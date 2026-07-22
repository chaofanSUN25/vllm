# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Pure-prefill TTFT benchmark for layer drop (core experiment).

Goal: measure the theoretical TTFT benefit of layer drop in isolation,
without decode-phase interference.

Workload:
  - One-shot batch: send N requests concurrently, wait for all to finish,
    then record TTFT. No continuous load.
  - Length distribution: 80% short requests (128 tokens) + 20% long stragglers
    (2048~4096 tokens).
  - Batch sizes: 4, 8, 16, 32.

Metrics:
  - TTFT P50 / P90 / P99
  - Success rate (requests that did not error or get truncated)
  - Per-request prompt length for diagnosis
  - Optional baseline vs layer-drop comparison when both URLs are given.
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any

import aiohttp


BATCH_SIZES = [4, 8, 16, 32]

SHORT_LEN = 128
LONG_MIN = 2048
LONG_MAX = 4096
LONG_RATIO = 0.2

MAX_TOKENS = 1  # Decode only 1 token to keep the measurement prefill-dominated


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


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


def make_prompt(length: int) -> str:
    """Return a synthetic prompt of roughly `length` tokens."""
    # Each "hello " is one token in most tokenizers.
    base = "hello "
    repeats = max(1, length)
    return base * repeats


def make_mixed_prompts(
    num_requests: int,
    short_len: int = SHORT_LEN,
    long_min: int = LONG_MIN,
    long_max: int = LONG_MAX,
    long_ratio: float = LONG_RATIO,
    seed: int = 42,
) -> list[tuple[str, int, str]]:
    random.seed(seed)
    num_long = max(1, int(num_requests * long_ratio)) if num_requests >= 5 else 0
    num_short = num_requests - num_long
    if num_short < 0:
        num_short = 0
        num_long = num_requests

    prompts: list[tuple[str, int, str]] = []
    for _ in range(num_short):
        prompts.append((make_prompt(short_len), short_len, "short"))
    for _ in range(num_long):
        length = random.randint(long_min, long_max)
        prompts.append((make_prompt(length), length, "long"))

    random.shuffle(prompts)
    return prompts


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    prompt_len: int,
    prompt_type: str,
    max_tokens: int,
    results: list[dict[str, Any]],
    idx: int,
) -> None:
    t_start = time.perf_counter()
    ttft_ms: float | None = None
    finish_reason: str | None = None
    num_tokens = 0
    try:
        async with session.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=aiohttp.ClientTimeout(total=300),
        ) as r:
            r.raise_for_status()
            first = True
            async for line in r.content:
                if not line:
                    continue
                if first:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    first = False
                num_tokens += 1
        e2e_ms = (time.perf_counter() - t_start) * 1000.0
        results[idx] = {
            "idx": idx,
            "prompt_len": prompt_len,
            "prompt_type": prompt_type,
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
            "num_tokens": num_tokens,
            "success": True,
        }
    except Exception as e:
        results[idx] = {
            "idx": idx,
            "prompt_len": prompt_len,
            "prompt_type": prompt_type,
            "error": str(e),
            "success": False,
        }


async def run_one_batch_size(
    url: str,
    model: str,
    prompts: list[tuple[str, int, str]],
    max_tokens: int,
    batch_size: int,
) -> dict[str, Any]:
    # Slice the shared trace to exactly batch_size.
    batch = prompts[:batch_size]
    results: list[dict[str, Any]] = [None] * batch_size

    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as session:
        tasks = [
            asyncio.create_task(
                send_request(
                    session,
                    url,
                    model,
                    prompt,
                    prompt_len,
                    prompt_type,
                    max_tokens,
                    results,
                    i,
                )
            )
            for i, (prompt, prompt_len, prompt_type) in enumerate(batch)
        ]
        await asyncio.gather(*tasks)
    total_duration_s = time.perf_counter() - t0

    ok = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    short_ttfts = [
        r["ttft_ms"]
        for r in ok
        if r["prompt_type"] == "short" and r["ttft_ms"] is not None
    ]
    long_ttfts = [
        r["ttft_ms"]
        for r in ok
        if r["prompt_type"] == "long" and r["ttft_ms"] is not None
    ]

    return {
        "batch_size": batch_size,
        "num_prompts": len(batch),
        "success": len(ok),
        "failed": len(failed),
        "success_rate_pct": len(ok) / len(batch) * 100 if batch else 0.0,
        "total_duration_s": total_duration_s,
        "ttft_ms": latency_summary(ttfts),
        "ttft_short_ms": latency_summary(short_ttfts),
        "ttft_long_ms": latency_summary(long_ttfts),
        "per_request": results,
    }


async def benchmark_url(
    name: str,
    url: str,
    model: str,
    trace: list[tuple[str, int, str]],
    max_tokens: int,
) -> dict[str, Any]:
    print(f"\n=== Running {name} on {url} ===")
    results = []
    for bs in BATCH_SIZES:
        print(f"  batch_size={bs} ...")
        exp = await run_one_batch_size(url, model, trace, max_tokens, bs)
        results.append(exp)
        print(
            f"    success_rate={exp['success_rate_pct']:.1f}% "
            f"ttft_p50={exp['ttft_ms']['p50_ms']:.2f}ms "
            f"ttft_p90={exp['ttft_ms']['p90_ms']:.2f}ms "
            f"ttft_p99={exp['ttft_ms']['p99_ms']:.2f}ms"
        )
    return {"name": name, "url": url, "batch_results": results}


def print_comparison(baseline: dict[str, Any], layer_drop: dict[str, Any]) -> None:
    print("\n=== Baseline vs Layer Drop TTFT Speedup ===")
    header = f"{'bs':>5} {'p50_speedup':>12} {'p90_speedup':>12} {'p99_speedup':>12}"
    print(header)
    print("-" * len(header))
    for b, l in zip(baseline["batch_results"], layer_drop["batch_results"]):
        def speedup(key: str) -> float:
            base_v = b["ttft_ms"][key]
            ld_v = l["ttft_ms"][key]
            return base_v / ld_v if ld_v else 0.0

        print(
            f"{b['batch_size']:>5} "
            f"{speedup('p50_ms'):>12.3f} "
            f"{speedup('p90_ms'):>12.3f} "
            f"{speedup('p99_ms'):>12.3f}"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", default=None)
    parser.add_argument("--layer-drop-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/layer_drop_prefill_ttft.json")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                        help="Number of decode tokens; use 1 for pure prefill.")
    parser.add_argument("--short-len", type=int, default=SHORT_LEN)
    parser.add_argument("--long-min", type=int, default=LONG_MIN)
    parser.add_argument("--long-max", type=int, default=LONG_MAX)
    parser.add_argument("--long-ratio", type=float, default=LONG_RATIO)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=BATCH_SIZES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.baseline_url and not args.layer_drop_url:
        parser.error("At least one of --baseline-url or --layer-drop-url "
                     "must be provided.")

    # Single trace used by all batch sizes, exactly like the ratio_batch script.
    max_bs = max(args.batch_sizes)
    trace = make_mixed_prompts(
        max_bs,
        args.short_len,
        args.long_min,
        args.long_max,
        args.long_ratio,
        args.seed,
    )
    print(
        f"Generated trace: {len(trace)} requests "
        f"({int(args.long_ratio * 100)}% long, seed={args.seed})"
    )

    output: dict[str, Any] = {
        "config": {
            "model": args.model,
            "max_tokens": args.max_tokens,
            "short_len": args.short_len,
            "long_min": args.long_min,
            "long_max": args.long_max,
            "long_ratio": args.long_ratio,
            "batch_sizes": args.batch_sizes,
            "seed": args.seed,
        },
        "experiments": [],
    }

    if args.baseline_url:
        exp = await benchmark_url(
            "baseline", args.baseline_url, args.model, trace, args.max_tokens
        )
        output["experiments"].append(exp)
    if args.layer_drop_url:
        exp = await benchmark_url(
            "layer_drop", args.layer_drop_url, args.model, trace, args.max_tokens
        )
        output["experiments"].append(exp)

    if len(output["experiments"]) == 2:
        baseline = output["experiments"][0]
        layer_drop = output["experiments"][1]
        output["comparison"] = {
            "speedups": [
                {
                    "batch_size": b["batch_size"],
                    "p50_speedup": b["ttft_ms"]["p50_ms"]
                    / layer_drop["batch_results"][i]["ttft_ms"]["p50_ms"]
                    if layer_drop["batch_results"][i]["ttft_ms"]["p50_ms"]
                    else 0.0,
                    "p90_speedup": b["ttft_ms"]["p90_ms"]
                    / layer_drop["batch_results"][i]["ttft_ms"]["p90_ms"]
                    if layer_drop["batch_results"][i]["ttft_ms"]["p90_ms"]
                    else 0.0,
                    "p99_speedup": b["ttft_ms"]["p99_ms"]
                    / layer_drop["batch_results"][i]["ttft_ms"]["p99_ms"]
                    if layer_drop["batch_results"][i]["ttft_ms"]["p99_ms"]
                    else 0.0,
                }
                for i, b in enumerate(baseline["batch_results"])
            ]
        }
        print_comparison(baseline, layer_drop)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote pure-prefill TTFT benchmark results to {out}")


if __name__ == "__main__":
    asyncio.run(main())
