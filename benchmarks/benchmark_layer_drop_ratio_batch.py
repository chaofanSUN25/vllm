# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Batch-size sensitivity benchmark for layer drop evaluation.

Tests layer drop performance across 9 batch sizes using a fixed request trace.
All batch sizes share the same trace (same length distribution), only concurrency
level differs. This ensures fair comparison across batch sizes.

Key design:
- Single trace of mixed prompts (30% long, 70% short) generated once
- All batch sizes use identical trace, only inflight concurrency varies
- Uses asyncio + aiohttp with Semaphore for precise concurrency control
- Wave-based sending ensures proper batching on server side
"""

import argparse
import asyncio
import json
import random
import statistics
import time
from pathlib import Path
from typing import Any, Tuple, List

import aiohttp


# Fixed batch sizes to test
BATCH_SIZES = [4, 6, 8, 16, 24, 32, 48, 64, 128]


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


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    results: list[dict[str, Any]],
    idx: int,
    prompt_type: str = "short",
) -> None:
    """Send a single streaming request and store result in results[idx]."""
    t_start = time.perf_counter()
    ttft_ms = None
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
                if line:
                    if first:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                        first = False
                    num_tokens += 1
        e2e_ms = (time.perf_counter() - t_start) * 1000.0
        results[idx] = {"idx": idx, "prompt_type": prompt_type, 
                        "ttft_ms": ttft_ms, "e2e_ms": e2e_ms, "num_tokens": num_tokens}
    except Exception as e:
        results[idx] = {"idx": idx, "error": str(e)}


async def run_batch_experiment(
    url: str,
    model: str,
    prompts: list[Tuple[str, str]],
    max_tokens: int,
    batch_size: int,
    slo_ttft_ms: int,
    slo_e2e_ms: int,
) -> dict[str, Any]:
    """Run experiment for a single batch size.
    
    Sends requests in waves of exactly batch_size, ensuring the server
    processes them as proper batches. Uses asyncio with Semaphore to control
    concurrency precisely.
    
    All batch sizes use the SAME trace of prompts, only the concurrency level
    (wave size) differs. This ensures fair comparison.
    """
    results: list[dict[str, Any]] = [None] * len(prompts)
    t0 = time.perf_counter()
    
    async with aiohttp.ClientSession() as session:
        # Send requests in waves of batch_size
        for wave_start in range(0, len(prompts), batch_size):
            wave = prompts[wave_start:wave_start + batch_size]
            
            # Submit all requests in this wave concurrently
            async with asyncio.Semaphore(len(wave)):
                tasks = []
                for i, (prompt, prompt_type) in enumerate(wave):
                    global_idx = wave_start + i
                    task = asyncio.create_task(
                        send_request(session, url, model, prompt, max_tokens, results,
                                    global_idx, prompt_type)
                    )
                    tasks.append(task)
                
                # Wait for all in wave to complete before sending next wave
                await asyncio.gather(*tasks)
    
    total_duration_s = time.perf_counter() - t0
    
    ok = [r for r in results if r["error"] is None]
    failed = [r for r in results if r["error"] is not None]
    
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok]
    
    # SLO attainment
    slo_ttft_met = sum(1 for r in ok if r["ttft_ms"] is not None and r["ttft_ms"] <= slo_ttft_ms)
    slo_e2e_met = sum(1 for r in ok if r["e2e_ms"] <= slo_e2e_ms)
    
    # Per-request SLO status
    for r in ok:
        r["slo_ttft_met"] = r["ttft_ms"] is not None and r["ttft_ms"] <= slo_ttft_ms
        r["slo_e2e_met"] = r["e2e_ms"] <= slo_e2e_ms
    
    return {
        "batch_size": batch_size,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "success": len(ok),
        "failed": len(failed),
        "total_duration_s": total_duration_s,
        "throughput_rps": len(ok) / total_duration_s if total_duration_s else 0.0,
        "ttft_ms": latency_summary(ttfts),
        "e2e_ms": latency_summary(e2es),
        "slo_ttft_ms": slo_ttft_ms,
        "slo_e2e_ms": slo_e2e_ms,
        "slo_ttft_attainment_pct": (slo_ttft_met / len(ok) * 100) if ok else 0.0,
        "slo_e2e_attainment_pct": (slo_e2e_met / len(ok) * 100) if ok else 0.0,
        "per_request": results,
    }


def make_mixed_prompts(
    num_requests: int,
    short_len: int = 8,
    long_len: int = 512,
    long_ratio: float = 0.3,
    seed: int = 42,
) -> list[tuple[str, str]]:
    random.seed(seed)
    short = "hello " * short_len
    long = (
        "In the year 2147, humanity discovered a way to travel faster than light. "
        "The first expedition to the Andromeda galaxy was planned for over a decade "
        "and involved thousands of scientists from every nation on Earth. "
    ) * (long_len // 30)
    num_long = int(num_requests * long_ratio)
    num_short = num_requests - num_long
    
    prompts = [(short, "short") for _ in range(num_short)] + [(long, "long") for _ in range(num_long)]
    random.shuffle(prompts)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-size sensitivity benchmark for layer drop"
    )
    parser.add_argument("--url", required=True, help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", required=True, help="Output JSON file path")
    parser.add_argument("--drop-ratio", type=float, default=0.0,
                        help="Drop ratio (metadata tag, server-determined)")
    
    # Benchmark parameters
    parser.add_argument("--num-requests", type=int, default=128,
                        help="Total requests per batch size (same for all batch sizes)")
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--long-ratio", type=float, default=0.3,
                        help="Ratio of long prompts in the trace")
    parser.add_argument("--short-len", type=int, default=8,
                        help="Approximate length of short prompts")
    parser.add_argument("--long-len", type=int, default=512,
                        help="Approximate length of long prompts")
    parser.add_argument("--slo-ttft-ms", type=int, default=500)
    parser.add_argument("--slo-e2e-ms", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Generate ONE fixed trace for ALL batch sizes
    # This ensures identical length distribution across all experiments
    trace = make_mixed_prompts(
        args.num_requests, args.short_len, args.long_len, args.long_ratio, args.seed
    )
    
    # Verify trace composition
    num_long = sum(1 for _, ptype in trace if ptype == "long")
    print(f"Generated fixed trace: {len(trace)} requests ({num_long} long, {len(trace)-num_long} short)")
    
    print(f"\nStarting batch-size sensitivity benchmark")
    print(f"  Server: {args.url}")
    print(f"  Model: {args.model}")
    print(f"  Drop ratio (server): {args.drop_ratio}")
    print(f"  Batch sizes: {BATCH_SIZES}")
    print(f"  Requests per experiment: {args.num_requests}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Output: {args.output}")
    
    # Run all 9 batch sizes using the SAME trace
    results: dict[str, Any] = {
        "config": {
            "url": args.url,
            "model": args.model,
            "drop_ratio": args.drop_ratio,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
            "long_ratio": args.long_ratio,
            "short_len": args.short_len,
            "long_len": args.long_len,
            "slo_ttft_ms": args.slo_ttft_ms,
            "slo_e2e_ms": args.slo_e2e_ms,
            "seed": args.seed,
            "trace_composition": {
                "total": len(trace),
                "long": num_long,
                "short": len(trace) - num_long,
            },
        },
        "experiments": [],
    }
    
    for batch_size in BATCH_SIZES:
        print(f"\n--- Testing batch_size={batch_size} ---")
        print(f"  Using identical trace of {len(trace)} requests")
        
        exp = asyncio.run(run_batch_experiment(
            url=args.url,
            model=args.model,
            prompts=trace,  # SAME trace for ALL batch sizes
            max_tokens=args.max_tokens,
            batch_size=batch_size,
            slo_ttft_ms=args.slo_ttft_ms,
            slo_e2e_ms=args.slo_e2e_ms,
        ))
        
        results["experiments"].append(exp)
        
        num_waves = (len(trace) + batch_size - 1) // batch_size
        print(f"  Requests: {exp['num_prompts']} (waves: {num_waves})")
        print(f"  Throughput: {exp['throughput_rps']:.2f} rps")
        print(f"  TTFT: mean={exp['ttft_ms']['mean_ms']:.2f}ms, p90={exp['ttft_ms']['p90_ms']:.2f}ms")
        print(f"  E2E: mean={exp['e2e_ms']['mean_ms']:.2f}ms, p90={exp['e2e_ms']['p90_ms']:.2f}ms")
        print(f"  SLO: TTFT={exp['slo_ttft_attainment_pct']:.1f}%, E2E={exp['slo_e2e_attainment_pct']:.1f}%")
    
    # Write results
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults written to {out}")
    print(f"Total experiments: {len(results['experiments'])}")
    print(f"\nAll experiments used the SAME trace - fair comparison enabled!")


if __name__ == "__main__":
    main()
