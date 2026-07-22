# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Max drop ratio sweep for layer drop sweet-spot detection.

Configuration:
  - Fixed batch size = 16
  - Skewed distribution: 80% short (128 tokens) + 20% long stragglers
    (2048~4096 tokens)
  - Variable: max_drop_ratio = 0.0, 0.1, 0.2, 0.3, 0.5

Metrics:
  - TTFT P99 (tail latency)
  - Throughput (requests / total duration)
  - Success rate (too aggressive dropping may truncate too many requests)

The same trace is reused for every ratio so that only max_drop_ratio varies.
The trace is sent against a single layer-drop server; the caller is expected
to restart the server with a different VLLM_LAYER_DROP_MAX_RATIO between runs,
or to point --url at different servers.
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


BATCH_SIZE = 16
SHORT_LEN = 128
LONG_MIN = 2048
LONG_MAX = 4096
LONG_RATIO = 0.2
DROP_RATIOS = [0.0, 0.1, 0.2, 0.3, 0.5]
MAX_TOKENS = 1  # Keep decode-phase interference minimal


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
    rng = random.Random(seed)
    num_long = max(1, round(num_requests * long_ratio))
    num_long = min(num_long, num_requests)
    num_short = num_requests - num_long

    prompts: list[tuple[str, int, str]] = []
    for _ in range(num_long):
        length = rng.randint(long_min, long_max)
        prompts.append((make_prompt(length), length, "long"))
    for _ in range(num_short):
        prompts.append((make_prompt(short_len), short_len, "short"))

    rng.shuffle(prompts)
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
            async for raw in r.content:
                if not raw:
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


async def run_one_ratio(
    url: str,
    model: str,
    prompts: list[tuple[str, int, str]],
    max_tokens: int,
    drop_ratio: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = [None] * len(prompts)

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
            for i, (prompt, prompt_len, prompt_type) in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)
    total_duration_s = time.perf_counter() - t0

    ok = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok]

    return {
        "drop_ratio": drop_ratio,
        "batch_size": len(prompts),
        "num_long": sum(1 for _, _, t in prompts if t == "long"),
        "num_short": sum(1 for _, _, t in prompts if t == "short"),
        "success": len(ok),
        "failed": len(failed),
        "success_rate_pct": len(ok) / len(prompts) * 100 if prompts else 0.0,
        "total_duration_s": total_duration_s,
        "throughput_rps": len(ok) / total_duration_s if total_duration_s else 0.0,
        "ttft_ms": latency_summary(ttfts),
        "e2e_ms": latency_summary(e2es),
        "per_request": results,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True,
                        help="vLLM server URL (completions endpoint)")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/layer_drop_ratio_sweep.json")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--drop-ratios", type=float, nargs="+",
                        default=DROP_RATIOS)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--short-len", type=int, default=SHORT_LEN)
    parser.add_argument("--long-min", type=int, default=LONG_MIN)
    parser.add_argument("--long-max", type=int, default=LONG_MAX)
    parser.add_argument("--long-ratio", type=float, default=LONG_RATIO)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = make_mixed_prompts(
        args.batch_size,
        args.short_len,
        args.long_min,
        args.long_max,
        args.long_ratio,
        args.seed,
    )
    num_long = sum(1 for _, _, t in prompts if t == "long")
    print(
        f"Fixed trace: batch_size={args.batch_size}, "
        f"{num_long} long, {args.batch_size - num_long} short, "
        f"seed={args.seed}"
    )
    print(f"Sweeping drop ratios: {args.drop_ratios}")

    results = []
    for ratio in args.drop_ratios:
        print(f"\n  max_drop_ratio={ratio:.2f} ...")
        exp = await run_one_ratio(
            args.url, args.model, prompts, args.max_tokens, ratio
        )
        results.append(exp)
        print(
            f"    success_rate={exp['success_rate_pct']:.1f}% "
            f"throughput={exp['throughput_rps']:.2f} req/s "
            f"ttft_p99={exp['ttft_ms']['p99_ms']:.2f}ms "
            f"ttft_mean={exp['ttft_ms']['mean_ms']:.2f}ms"
        )

    print("\n=== Drop Ratio Sweep Summary ===")
    header = (
        f"{'ratio':>8} {'success%':>8} {'thrpt_rps':>10} "
        f"{'ttft_p99':>11} {'ttft_mean':>11}"
    )
    print(header)
    print("-" * len(header))
    for exp in results:
        print(
            f"{exp['drop_ratio']:>8.2f} "
            f"{exp['success_rate_pct']:>8.1f} "
            f"{exp['throughput_rps']:>10.2f} "
            f"{exp['ttft_ms']['p99_ms']:>11.2f} "
            f"{exp['ttft_ms']['mean_ms']:>11.2f}"
        )

    output: dict[str, Any] = {
        "config": {
            "url": args.url,
            "model": args.model,
            "batch_size": args.batch_size,
            "max_tokens": args.max_tokens,
            "short_len": args.short_len,
            "long_min": args.long_min,
            "long_max": args.long_max,
            "long_ratio": args.long_ratio,
            "seed": args.seed,
        },
        "results": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWrote drop-ratio sweep results to {out}")


if __name__ == "__main__":
    asyncio.run(main())
