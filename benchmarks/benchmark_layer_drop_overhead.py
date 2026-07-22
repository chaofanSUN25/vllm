# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-layer overhead benchmark for the three layer-drop data-movement steps.

Times one decoder layer invocation, mirroring
Qwen2Model._forward_layer_with_drop / LlamaModel._forward_layer_with_drop:

1. compact: LayerDropManager.compact_hidden_states + positions/residual
   gather.
2. update_metadata: LayerDropManager.update_metadata, via either the
   CommonAttentionMetadata or the FlashAttentionMetadata route.
3. scatter: zeros_like + index copy back to the original token layout
   (hidden_states and residual).

Wall-clock timing with an explicit sync around every repetition is used on
purpose: the measured code paths contain inherent GPU->CPU syncs (.item(),
.cpu()), so wall time is what shows up in end-to-end TTFT.

Experiments:
    A: seq_len=512 fixed, batch_size in [4, 8, 16, 32, 64, 128]
    B: batch_size=16 fixed, seq_len in [64, 128, 256, 512, 1024, 2048]
    C: total_tokens=8192 fixed,
       (batch_size, seq_len) in [(4, 2048), (16, 512), (64, 128), (128, 64)]
"""

import argparse
import gc
import json
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from vllm.model_executor.layers.layer_drop import LayerDropManager
from vllm.v1.attention.backend import CommonAttentionMetadata

try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    FLASH_AVAILABLE = True
except ImportError:
    FlashAttentionMetadata = None
    FLASH_AVAILABLE = False

EXPERIMENTS: dict[str, tuple[str, list[tuple[int, int]]]] = {
    "A": (
        "seq_len=512 fixed, sweep batch_size",
        [(bs, 512) for bs in [4, 8, 16, 32, 64, 128]],
    ),
    "B": (
        "batch_size=16 fixed, sweep seq_len",
        [(16, sl) for sl in [64, 128, 256, 512, 1024, 2048]],
    ),
    "C": (
        "total_tokens=8192 fixed",
        [(4, 2048), (16, 512), (64, 128), (128, 64)],
    ),
}

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def summarize(samples_ms: list[float]) -> dict[str, float]:
    return {
        "n": len(samples_ms),
        "mean_ms": statistics.mean(samples_ms) if samples_ms else 0.0,
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50_ms": percentile(samples_ms, 50),
        "p95_ms": percentile(samples_ms, 95),
        "min_ms": min(samples_ms) if samples_ms else 0.0,
        "max_ms": max(samples_ms) if samples_ms else 0.0,
    }


def make_keep_mask(
    num_reqs: int, drop_ratio: float, device: torch.device
) -> torch.Tensor:
    """Drop the last k requests.

    With uniform seq_lens the manager's drop scores tie and its deterministic
    index tie-break drops the highest indices, so this matches production
    masks while keeping the benchmark independent of the drop-decision path.
    """
    k = min(int(num_reqs * drop_ratio), num_reqs - 1)
    keep_mask = torch.ones(num_reqs, dtype=torch.bool, device=device)
    if k > 0:
        keep_mask[num_reqs - k :] = False
    return keep_mask


def build_common_metadata(
    num_reqs: int, seq_len: int, device: torch.device
) -> CommonAttentionMetadata:
    num_tokens = num_reqs * seq_len
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    return CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc.cpu(),
        seq_lens=torch.full(
            (num_reqs,), seq_len, dtype=torch.int32, device=device
        ),
        num_reqs=num_reqs,
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=torch.zeros(
            num_reqs, 16, dtype=torch.int32, device=device
        ),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device),
        causal=True,
        is_prefilling=torch.ones(num_reqs, dtype=torch.bool, device=device),
    )


def build_flash_metadata(num_reqs: int, seq_len: int, device: torch.device):
    num_tokens = num_reqs * seq_len
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    return FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=torch.full(
            (num_reqs,), seq_len, dtype=torch.int32, device=device
        ),
        block_table=torch.zeros(num_reqs, 16, dtype=torch.int32, device=device),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32, device=device),
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
    )


def time_op(fn, device: torch.device, warmup: int, repetitions: int):
    """Wall-clock ms per call, with a full sync around every repetition."""
    for _ in range(warmup):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize()
    samples: list[float] = []
    gc_was_enabled = gc.isenabled()
    if gc_was_enabled:
        gc.disable()
    try:
        for _ in range(repetitions):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1e3)
    finally:
        if gc_was_enabled:
            gc.enable()
    return samples


def run_config(
    manager: LayerDropManager,
    num_reqs: int,
    seq_len: int,
    backends: list[str],
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    num_tokens = num_reqs * seq_len
    hidden_states = torch.randn(
        num_tokens, args.hidden_size, dtype=dtype, device=device
    )
    residual = torch.randn_like(hidden_states)
    positions = torch.arange(num_tokens, device=device) % seq_len
    query_start_loc = torch.arange(
        0, (num_reqs + 1) * seq_len, seq_len, dtype=torch.int32, device=device
    )
    keep_mask = make_keep_mask(num_reqs, args.drop_ratio, device)
    num_dropped = int((~keep_mask).sum().item())

    compacted_hs, index_map, keep_indices = manager.compact_hidden_states(
        hidden_states, keep_mask, query_start_loc
    )
    compacted_residual = residual[keep_indices]

    def compact_step():
        _, _, kept = manager.compact_hidden_states(
            hidden_states, keep_mask, query_start_loc
        )
        positions[kept]
        residual[kept]

    def scatter_step():
        full_hs = torch.zeros_like(hidden_states)
        full_hs[keep_indices] = compacted_hs
        full_residual = torch.zeros_like(residual)
        full_residual[keep_indices] = compacted_residual

    result: dict[str, Any] = {
        "batch_size": num_reqs,
        "seq_len": seq_len,
        "num_tokens": num_tokens,
        "num_dropped": num_dropped,
        "kept_tokens": int(keep_indices.shape[0]),
        "backends": {},
    }
    for backend in backends:
        metadata = (
            build_common_metadata(num_reqs, seq_len, device)
            if backend == "common"
            else build_flash_metadata(num_reqs, seq_len, device)
        )

        def update_step():
            manager.update_metadata(metadata, keep_mask, index_map, keep_indices)

        compact_ms = time_op(compact_step, device, args.warmup, args.repetitions)
        update_ms = time_op(update_step, device, args.warmup, args.repetitions)
        scatter_ms = time_op(scatter_step, device, args.warmup, args.repetitions)
        result["backends"][backend] = {
            "compact_ms": summarize(compact_ms),
            "update_metadata_ms": summarize(update_ms),
            "scatter_ms": summarize(scatter_ms),
            "total_mean_ms": (
                statistics.mean(compact_ms)
                + statistics.mean(update_ms)
                + statistics.mean(scatter_ms)
            ),
        }
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def print_tables(exp_name: str, description: str, results: list[dict[str, Any]]):
    backends = sorted({b for r in results for b in r["backends"]})
    for backend in backends:
        print(f"\n=== Experiment {exp_name}: {description} "
              f"| metadata route: {backend} ===")
        header = (
            f"{'bs':>5} {'seq_len':>8} {'tokens':>8} {'dropped':>8} "
            f"{'compact_ms':>11} {'update_ms':>10} {'scatter_ms':>11} "
            f"{'total_ms':>9}"
        )
        print(header)
        print("-" * len(header))
        for r in results:
            b = r["backends"].get(backend)
            if b is None:
                continue
            print(
                f"{r['batch_size']:>5} {r['seq_len']:>8} {r['num_tokens']:>8} "
                f"{r['num_dropped']:>8} "
                f"{b['compact_ms']['mean_ms']:>11.4f} "
                f"{b['update_metadata_ms']['mean_ms']:>10.4f} "
                f"{b['scatter_ms']['mean_ms']:>11.4f} "
                f"{b['total_mean_ms']:>9.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--hidden-size", type=int, default=896,
                        help="Qwen2.5-0.5B hidden size")
    parser.add_argument("--drop-ratio", type=float, default=0.3)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repetitions", type=int, default=100)
    parser.add_argument("--metadata-backend",
                        choices=["common", "flash", "both"], default="both")
    parser.add_argument("--experiments", default="ABC",
                        help="Subset of experiments to run, e.g. 'AC'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="results/layer_drop_overhead.json")
    args = parser.parse_args()

    if args.metadata_backend == "both":
        backends = ["common", "flash"]
    else:
        backends = [args.metadata_backend]
    if "flash" in backends and not FLASH_AVAILABLE:
        print("WARNING: flash_attn backend unavailable, "
              "skipping the flash metadata route.")
        backends.remove("flash")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    manager = LayerDropManager(max_drop_ratio=args.drop_ratio, enabled=True)

    print(f"device={device} dtype={args.dtype} hidden_size={args.hidden_size} "
          f"drop_ratio={args.drop_ratio} warmup={args.warmup} "
          f"reps={args.repetitions} backends={backends}")

    output: dict[str, Any] = {
        "config": {
            "device": str(device),
            "dtype": args.dtype,
            "hidden_size": args.hidden_size,
            "drop_ratio": args.drop_ratio,
            "warmup": args.warmup,
            "repetitions": args.repetitions,
            "seed": args.seed,
            "note": "times are per dropped layer; multiply by the number of "
                    "dropped layers for full forward-pass overhead",
        },
        "experiments": {},
    }
    for name in args.experiments.upper():
        if name not in EXPERIMENTS:
            print(f"unknown experiment '{name}', skipping")
            continue
        description, configs = EXPERIMENTS[name]
        results = [
            run_config(manager, bs, sl, backends, args, device, dtype)
            for bs, sl in configs
        ]
        print_tables(name, description, results)
        output["experiments"][name] = {
            "description": description,
            "results": results,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote overhead benchmark results to {out}")


if __name__ == "__main__":
    main()
