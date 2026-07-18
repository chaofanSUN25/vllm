# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ablation study for layer drop policy hyperparameters.

Sweeps max_drop_ratio, length_ratio_threshold, priority_weight, and
length_weight, and records drop ratios, per-layer distributions, and the
composition of dropped requests.
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vllm.model_executor.layers.layer_drop import LayerDropManager


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_ablation(
    seq_lens: list[int],
    total_layers: int,
    max_drop_ratio: float,
    length_ratio_threshold: float,
    priority_weight: float,
    length_weight: float,
    device: torch.device,
) -> dict[str, Any]:
    num_reqs = len(seq_lens)
    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=device)
    priorities = torch.rand(num_reqs, device=device)
    is_prefilling = torch.ones(num_reqs, dtype=torch.bool, device=device)

    manager = LayerDropManager(
        max_drop_ratio=max_drop_ratio,
        length_ratio_threshold=length_ratio_threshold,
        priority_weight=priority_weight,
        length_weight=length_weight,
        enabled=True,
    )
    manager.precompute_layer_drop_masks(
        seq_lens=seq_lens_t,
        total_layers=total_layers,
        priorities=priorities,
        is_prefilling=is_prefilling,
    )

    final_indices = set(manager.get_dropped_req_indices())
    straggler_mask = manager._detect_stragglers(seq_lens_t, num_reqs)
    stragglers = set(straggler_mask.nonzero(as_tuple=False).flatten().tolist())

    per_layer = []
    for layer_idx in range(total_layers):
        mask = manager.get_drop_mask_for_layer(layer_idx)
        if mask is None:
            continue
        per_layer.append({
            "layer": layer_idx,
            "dropped": int(mask.sum().item()),
        })

    return {
        "max_drop_ratio": max_drop_ratio,
        "length_ratio_threshold": length_ratio_threshold,
        "priority_weight": priority_weight,
        "length_weight": length_weight,
        "drop_count": len(final_indices),
        "drop_ratio": len(final_indices) / num_reqs,
        "straggler_drop_count": len(final_indices & stragglers),
        "non_straggler_drop_count": len(final_indices - stragglers),
        "per_layer": per_layer,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="results/layer_drop_ablation.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-layers", type=int, default=24)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    # Synthetic workload: 16 requests with a clear straggler tail.
    seq_lens = [32, 48, 64, 80, 96, 128, 256, 512,
                32, 48, 64, 80, 96, 128, 256, 2048]

    configs = []
    for max_drop_ratio in [0.05, 0.1, 0.2, 0.3]:
        for length_ratio_threshold in [1.5, 2.0, 3.0]:
            configs.append({
                "max_drop_ratio": max_drop_ratio,
                "length_ratio_threshold": length_ratio_threshold,
                "priority_weight": 1.0,
                "length_weight": 1.0,
            })

    for priority_weight in [0.0, 0.5, 1.0, 2.0]:
        for length_weight in [0.0, 0.5, 1.0, 2.0]:
            configs.append({
                "max_drop_ratio": 0.1,
                "length_ratio_threshold": 2.0,
                "priority_weight": priority_weight,
                "length_weight": length_weight,
            })

    results: list[dict[str, Any]] = []
    for cfg in configs:
        print(f"Running ablation: {cfg}")
        results.append(run_ablation(seq_lens, args.total_layers, **cfg,
                                    device=device))

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump({
            "config": {
                "device": str(device),
                "seed": args.seed,
                "total_layers": args.total_layers,
                "seq_lens": seq_lens,
            },
            "results": results,
        }, f, indent=2)
    print(f"Wrote ablation results to {out}")


if __name__ == "__main__":
    main()