# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-batch drop and latency-impact analysis for layer drop.

For each synthetic batch, records how many requests are dropped, which layers
drop them, and estimates the latency reduction brought to the batch.
"""

import argparse
import contextlib
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import (
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layer_drop import LayerDropManager


def init_vllm_distributed() -> None:
    """Initialize a minimal TP group for single-process benchmarks.

    LayerDropManager calls get_tensor_model_parallel_world_size() to
    synchronize drop masks, so the parallel state must be initialized.
    """
    import torch.distributed as dist

    if dist.is_initialized():
        return

    fd, temp_file = tempfile.mkstemp()
    os.close(fd)
    try:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        with set_current_vllm_config(VllmConfig()):
            init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method=f"file://{temp_file}",
                local_rank=0,
                backend=backend,
            )
            initialize_model_parallel(1, 1)
    finally:
        with contextlib.suppress(OSError):
            os.unlink(temp_file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def estimate_batch_latency_ms(
    seq_lens: torch.Tensor,
    hidden_size: int,
    total_layers: int,
    drop_mask: torch.Tensor,
) -> float:
    """Rough FLOPs-based latency proxy: proportional to kept tokens * layers."""
    kept_tokens = int(seq_lens[~drop_mask].sum().item())
    total_tokens = int(seq_lens.sum().item())
    if kept_tokens == 0:
        return 0.0
    # Normalize to arbitrary units; we only need relative speedup.
    base = total_tokens * total_layers * hidden_size * 1e-5
    return base * kept_tokens / total_tokens


def analyze_batch(
    manager: LayerDropManager,
    seq_lens: list[int],
    total_layers: int,
    hidden_size: int,
    device: torch.device,
) -> dict[str, Any]:
    num_reqs = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
    priorities = torch.rand(num_reqs, device=device)
    is_prefilling = torch.ones(num_reqs, dtype=torch.bool, device=device)

    manager.reset()
    t0 = time.perf_counter()
    manager.precompute_layer_drop_masks(
        seq_lens=seq_lens_t,
        total_layers=total_layers,
        priorities=priorities,
        is_prefilling=is_prefilling,
    )
    decision_ms = (time.perf_counter() - t0) * 1000.0

    final_mask = manager.final_drop_mask
    if final_mask is None:
        final_mask = torch.zeros(num_reqs, dtype=torch.bool, device=device)

    baseline_latency = estimate_batch_latency_ms(
        seq_lens_t, hidden_size, total_layers,
        torch.zeros(num_reqs, dtype=torch.bool, device=device))
    drop_latency = estimate_batch_latency_ms(seq_lens_t, hidden_size,
                                             total_layers, final_mask)

    per_layer = []
    for layer_idx in range(total_layers):
        mask = manager.get_drop_mask_for_layer(layer_idx)
        if mask is None:
            continue
        newly = int((mask & ~(
            manager.get_drop_mask_for_layer(layer_idx - 1)
            if layer_idx > 0 else torch.zeros(num_reqs, dtype=torch.bool,
                                              device=device)
        )).sum().item())
        per_layer.append({
            "layer": layer_idx,
            "cumulative_dropped": int(mask.sum().item()),
            "newly_dropped": newly,
        })

    dropped_indices = manager.get_dropped_req_indices()
    dropped_seq_lens = [int(seq_lens[i]) for i in dropped_indices]

    return {
        "num_reqs": num_reqs,
        "seq_lens": seq_lens,
        "dropped_indices": dropped_indices,
        "dropped_seq_lens": dropped_seq_lens,
        "drop_count": len(dropped_indices),
        "drop_ratio": len(dropped_indices) / num_reqs,
        "decision_ms": decision_ms,
        "baseline_latency_ms": baseline_latency,
        "drop_latency_ms": drop_latency,
        "latency_reduction_ms": baseline_latency - drop_latency,
        "speedup": baseline_latency / drop_latency if drop_latency > 0 else 1.0,
        "per_layer": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/layer_drop_batch.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-layers", type=int, default=24)
    parser.add_argument("--hidden-size", type=int, default=4096)
    args = parser.parse_args()

    init_vllm_distributed()

    set_seed(args.seed)
    device = torch.device(args.device)

    manager = LayerDropManager(
        max_drop_ratio=0.1,
        length_ratio_threshold=2.0,
        priority_weight=1.0,
        length_weight=1.0,
        enabled=True,
    )

    workloads = [
        {"name": "uniform_short", "seq_lens": [32] * 16},
        {"name": "uniform_long", "seq_lens": [1024] * 16},
        {"name": "mixed_balanced", "seq_lens": [64] * 8 + [512] * 8},
        {"name": "one_straggler", "seq_lens": [32] * 15 + [2048]},
        {"name": "heavy_tail", "seq_lens": [32, 64, 128, 256, 512,
                                             1024, 1536, 2048]},
    ]

    results: dict[str, Any] = {
        "config": {
            "device": str(device),
            "seed": args.seed,
            "total_layers": args.total_layers,
            "hidden_size": args.hidden_size,
        },
        "batches": [],
    }

    for workload in workloads:
        print(f"Analyzing batch: {workload['name']}")
        results["batches"].append(
            analyze_batch(manager, workload["seq_lens"], args.total_layers,
                          args.hidden_size, device))

    # Aggregate
    speedups = [b["speedup"] for b in results["batches"]]
    reductions = [b["latency_reduction_ms"] for b in results["batches"]]
    drop_ratios = [b["drop_ratio"] for b in results["batches"]]

    results["summary"] = {
        "avg_speedup": float(np.mean(speedups)),
        "max_speedup": float(np.max(speedups)),
        "avg_latency_reduction_ms": float(np.mean(reductions)),
        "avg_drop_ratio": float(np.mean(drop_ratios)),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote batch analysis results to {out}")


if __name__ == "__main__":
    main()