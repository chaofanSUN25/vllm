# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end serving benchmark for layer drop.

Measures TTFT, end-to-end latency, throughput, SLO attainment, and tail
latencies (P90/P99) for a baseline server and a layer-drop-enabled server.
"""

import argparse
import json
import random
import statistics
import threading
import time
from pathlib import Path
from typing import Any

import requests


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


def send_request(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    results: list[dict[str, Any]],
    idx: int,
) -> None:
    t_start = time.perf_counter()
    ttft_ms = None
    try:
        with requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "stream": True,
            },
            stream=True,
            timeout=300,
        ) as r:
            r.raise_for_status()
            first = True
            for line in r.iter_lines():
                if line:
                    if first:
                        ttft_ms = (time.perf_counter() - t_start) * 1000.0
                        first = False
        e2e_ms = (time.perf_counter() - t_start) * 1000.0
        results.append({"idx": idx, "ttft_ms": ttft_ms, "e2e_ms": e2e_ms})
    except Exception as e:
        results.append({"idx": idx, "error": str(e)})


def run_serving_experiment(
    url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    threads = []
    t0 = time.perf_counter()
    for i, prompt in enumerate(prompts):
        t = threading.Thread(
            target=send_request,
            args=(url, model, prompt, max_tokens, results, i),
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()
    total_duration_s = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok]

    return {
        "url": url,
        "num_prompts": len(prompts),
        "max_tokens": max_tokens,
        "success": len(ok),
        "failed": len(prompts) - len(ok),
        "total_duration_s": total_duration_s,
        "throughput_rps": len(ok) / total_duration_s if total_duration_s else 0.0,
        "ttft_ms": latency_summary(ttfts),
        "e2e_ms": latency_summary(e2es),
    }


def make_mixed_prompts(
    num_short: int,
    num_long: int,
    short_len: int = 8,
    long_len: int = 512,
) -> list[str]:
    short = "hello " * short_len
    long = (
        "In the year 2147, humanity discovered a way to travel faster than light. "
        "The first expedition to the Andromeda galaxy was planned for over a decade "
        "and involved thousands of scientists from every nation on Earth. "
    ) * (long_len // 30)
    prompts = [short for _ in range(num_short)] + [long for _ in range(num_long)]
    random.shuffle(prompts)
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url",
                        default="http://localhost:8000/v1/completions")
    parser.add_argument("--layer-drop-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output", default="results/layer_drop_serving.json")
    parser.add_argument("--num-short", type=int, default=18)
    parser.add_argument("--num-long", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    prompts = make_mixed_prompts(args.num_short, args.num_long)

    urls = [("baseline", args.baseline_url)]
    if args.layer_drop_url:
        urls.append(("layer_drop", args.layer_drop_url))

    results: dict[str, Any] = {
        "config": {
            "model": args.model,
            "num_short": args.num_short,
            "num_long": args.num_long,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "experiments": [],
    }

    for name, url in urls:
        print(f"Running {name} benchmark against {url} ...")
        exp = run_serving_experiment(url, args.model, prompts, args.max_tokens)
        exp["name"] = name
        results["experiments"].append(exp)

    if len(results["experiments"]) == 2:
        base = results["experiments"][0]
        ld = results["experiments"][1]
        results["comparison"] = {
            "ttft_speedup": base["ttft_ms"]["mean_ms"] / ld["ttft_ms"]["mean_ms"]
            if ld["ttft_ms"]["mean_ms"] else 0.0,
            "e2e_speedup": base["e2e_ms"]["mean_ms"] / ld["e2e_ms"]["mean_ms"]
            if ld["e2e_ms"]["mean_ms"] else 0.0,
            "throughput_ratio": ld["throughput_rps"] / base["throughput_rps"]
            if base["throughput_rps"] else 0.0,
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote serving benchmark results to {out}")


if __name__ == "__main__":
    main()