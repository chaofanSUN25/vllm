# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Component-level performance benchmark for layer drop.

Measures per-layer drop counts, cumulative drop ratios, and the latency of
drop decision, hidden-state compaction, and metadata updates.
"""

import argparse
import contextlib
import json
import os
import random
import statistics
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.distributed.parallel_state import (
    cleanup_dist_env_and_memory,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.layer_drop import LayerDropManager
from vllm.v1.attention.backend import CommonAttentionMetadata


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


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def latency_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "n": len(samples_ms),
        "mean_ms": statistics.mean(samples_ms) if samples_ms else 0.0,
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p50_ms": percentile(samples_ms, 50),
        "p90_ms": percentile(samples_ms, 90),
        "p99_ms": percentile(samples_ms, 99),
        "min_ms": min(samples_ms) if samples_ms else 0.0,
        "max_ms": max(samples_ms) if samples_ms else 0.0,
    }


def make_metadata(
    seq_lens: list[int],
    query_lens: list[int],
    device: torch.device,
) -> CommonAttentionMetadata:
    query_start_loc = [0]
    for q in query_lens:
        query_start_loc.append(query_start_loc[-1] + q)
    num_tokens = query_start_loc[-1]
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32,
                                     device=device),
        query_start_loc_cpu=torch.tensor(query_start_loc, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=device),
        num_reqs=len(seq_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens) if query_lens else 0,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        block_table_tensor=torch.zeros(len(seq_lens), 16, dtype=torch.int32,
                                       device=device),
        slot_mapping=torch.arange(num_tokens, dtype=torch.int64, device=device),
        causal=True,
        is_prefilling=torch.ones(len(seq_lens), dtype=torch.bool, device=device),
    )


def run_drop_count_experiment(
    manager: LayerDropManager,
    num_reqs: int,
    total_layers: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    seq_lens = torch.randint(8, 1024, (num_reqs,), device=device).long()
    priorities = torch.rand(num_reqs, device=device)
    is_prefilling = torch.ones(num_reqs, dtype=torch.bool, device=device)

    manager.reset()
    manager.precompute_layer_drop_masks(
        seq_lens=seq_lens,
        total_layers=total_layers,
        priorities=priorities,
        is_prefilling=is_prefilling,
    )

    per_layer = []
    for layer_idx in range(total_layers):
        mask = manager.get_drop_mask_for_layer(layer_idx)
        if mask is None:
            break
        per_layer.append({
            "layer": layer_idx,
            "dropped": int(mask.sum().item()),
            "drop_ratio": float(mask.sum().item()) / num_reqs,
        })

    final_dropped = manager.get_dropped_req_indices()
    return {
        "num_reqs": num_reqs,
        "total_layers": total_layers,
        "final_dropped": len(final_dropped),
        "final_drop_ratio": len(final_dropped) / num_reqs,
        "per_layer": per_layer,
    }


def run_latency_experiment(
    manager: LayerDropManager,
    num_reqs: int,
    total_layers: int,
    hidden_size: int,
    device: torch.device,
    repetitions: int,
    seed: int,
) -> dict[str, Any]:
    set_seed(seed)
    seq_lens = torch.randint(16, 512, (num_reqs,), device=device).long()
    query_lens = seq_lens  # prefill
    query_start_loc = [0]
    for q in query_lens.tolist():
        query_start_loc.append(query_start_loc[-1] + q)
    query_start_loc_t = torch.tensor(query_start_loc, dtype=torch.int32,
                                     device=device)
    priorities = torch.rand(num_reqs, device=device)
    is_prefilling = torch.ones(num_reqs, dtype=torch.bool, device=device)
    num_tokens = query_start_loc[-1]
    hidden_states = torch.randn(num_tokens, hidden_size, device=device)

    precompute_times = []
    for _ in range(repetitions):
        manager.reset()
        t0 = now_ms()
        manager.precompute_layer_drop_masks(
            seq_lens=seq_lens,
            total_layers=total_layers,
            priorities=priorities,
            is_prefilling=is_prefilling,
        )
        precompute_times.append(now_ms() - t0)

    keep_mask = ~manager.get_drop_mask_for_layer(0)
    if keep_mask is None or not keep_mask.any():
        keep_mask = torch.ones(num_reqs, dtype=torch.bool, device=device)

    compact_times = []
    for _ in range(repetitions):
        t0 = now_ms()
        manager.compact_hidden_states(hidden_states, keep_mask, query_start_loc_t)
        compact_times.append(now_ms() - t0)

    metadata = make_metadata(seq_lens.tolist(), query_lens.tolist(), device)
    _, index_map, keep_indices = manager.compact_hidden_states(
        hidden_states, keep_mask, query_start_loc_t)

    update_times = []
    for _ in range(repetitions):
        t0 = now_ms()
        manager.update_metadata(metadata, keep_mask, index_map, keep_indices)
        update_times.append(now_ms() - t0)

    return {
        "num_reqs": num_reqs,
        "total_layers": total_layers,
        "hidden_size": hidden_size,
        "num_tokens": num_tokens,
        "precompute_ms": latency_summary(precompute_times),
        "compact_ms": latency_summary(compact_times),
        "update_metadata_ms": latency_summary(update_times),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/layer_drop_component.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=100)
    args = parser.parse_args()

    init_vllm_distributed()

    device = torch.device(args.device)
    results: dict[str, Any] = {
        "config": {
            "device": str(device),
            "seed": args.seed,
            "repetitions": args.repetitions,
        },
        "drop_count": [],
        "latency": [],
    }

    manager = LayerDropManager(
        max_drop_ratio=0.1,
        length_ratio_threshold=2.0,
        priority_weight=1.0,
        length_weight=1.0,
        enabled=True,
    )

    for num_reqs in [4, 8, 16, 32, 64]:
        for total_layers in [16, 24, 32]:
            results["drop_count"].append(run_drop_count_experiment(
                manager, num_reqs, total_layers, device, args.seed))
            results["latency"].append(run_latency_experiment(
                manager, num_reqs, total_layers, 4096, device,
                args.repetitions, args.seed))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote component benchmark results to {out}")
    cleanup_dist_env_and_memory()


if __name__ == "__main__":
    main()