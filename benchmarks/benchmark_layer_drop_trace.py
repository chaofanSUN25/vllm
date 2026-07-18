# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Real-workload benchmark using ShareGPT-style traces.

Loads prompts from a JSONL file, filters by length, samples a request stream,
and dispatches them according to a Poisson arrival process. Measures TTFT,
end-to-end latency, throughput, SLO attainment, and tail latencies.
"""

import argparse
import json
import random
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    k = (len(sorted_values) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(sorted_values) - 1)
    return sorted_values[f] + (sorted_values[c] - sorted_values[f]) * (k - f)


def load_sharegpt_prompts(path: Path, min_tokens: int,
                          max_tokens: int) -> list[str]:
    prompts: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            conv = data.get("conversations", [])
            for turn in conv:
                if turn.get("from") == "human":
                    text = turn.get("value", "")
                    length = len(text.split())
                    if min_tokens <= length <= max_tokens:
                        prompts.append(text)
                        break
    return prompts


def send_streaming(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    results: list[dict[str, Any]],
    idx: int,
) -> None:
    t_start = time.perf_counter()
    ttft_ms: float | None = None
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
            for _ in r.iter_lines():
                if first:
                    ttft_ms = (time.perf_counter() - t_start) * 1000.0
                    first = False
        e2e_ms = (time.perf_counter() - t_start) * 1000.0
        results.append({
            "idx": idx,
            "prompt_len": len(prompt.split()),
            "ttft_ms": ttft_ms,
            "e2e_ms": e2e_ms,
        })
    except Exception as e:
        results.append({"idx": idx, "error": str(e)})


def run_trace(
    url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    arrival_rate_rps: float,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    threads: list[threading.Thread] = []
    t0 = time.perf_counter()

    for i, prompt in enumerate(prompts):
        delay = random.expovariate(arrival_rate_rps) if arrival_rate_rps > 0 else 0
        time.sleep(delay)
        t = threading.Thread(
            target=send_streaming,
            args=(url, model, prompt, max_tokens, results, i),
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    total_s = time.perf_counter() - t0

    ok = [r for r in results if "error" not in r]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]
    e2es = [r["e2e_ms"] for r in ok]
    prompt_lens = [r["prompt_len"] for r in ok]

    slo_ms = 1000.0
    slo_attainment = sum(1 for v in e2es if v <= slo_ms) / len(e2es) if e2es else 0

    return {
        "url": url,
        "arrival_rate_rps": arrival_rate_rps,
        "num_prompts": len(prompts),
        "success": len(ok),
        "failed": len(prompts) - len(ok),
        "total_duration_s": total_s,
        "throughput_rps": len(ok) / total_s if total_s else 0.0,
        "mean_prompt_len": float(np.mean(prompt_lens)) if prompt_lens else 0.0,
        "ttft_ms": {
            "mean": float(np.mean(ttfts)) if ttfts else 0.0,
            "p50": percentile(ttfts, 50),
            "p90": percentile(ttfts, 90),
            "p99": percentile(ttfts, 99),
        },
        "e2e_ms": {
            "mean": float(np.mean(e2es)) if e2es else 0.0,
            "p50": percentile(e2es, 50),
            "p90": percentile(e2es, 90),
            "p99": percentile(e2es, 99),
        },
        "slo_attainment": slo_attainment,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-url",
                        default="http://localhost:8000/v1/completions")
    parser.add_argument("--layer-drop-url", default=None)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--dataset", default="ShareGPT_V3_unfiltered_cleaned_split.jsonl")
    parser.add_argument("--output", default="results/layer_drop_trace.json")
    parser.add_argument("--num-prompts", type=int, default=100)
    parser.add_argument("--min-prompt-len", type=int, default=16)
    parser.add_argument("--max-prompt-len", type=int, default=1024)
    parser.add_argument("--arrival-rate-rps", type=float, default=2.0)
    parser.add_argument("--max-tokens", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_path = Path(args.dataset)
    if dataset_path.exists():
        prompts = load_sharegpt_prompts(dataset_path, args.min_prompt_len,
                                          args.max_prompt_len)
    else:
        print(f"Dataset {dataset_path} not found, using synthetic prompts.")
        prompts = [
            "hello " * random.randint(args.min_prompt_len,
                                      args.max_prompt_len)
            for _ in range(args.num_prompts)
        ]

    if len(prompts) < args.num_prompts:
        prompts = (prompts * ((args.num_prompts // len(prompts)) + 1)
                   )[:args.num_prompts]
    else:
        prompts = random.sample(prompts, args.num_prompts)

    urls = [("baseline", args.baseline_url)]
    if args.layer_drop_url:
        urls.append(("layer_drop", args.layer_drop_url))

    output: dict[str, Any] = {
        "config": {
            "model": args.model,
            "dataset": str(dataset_path),
            "num_prompts": args.num_prompts,
            "min_prompt_len": args.min_prompt_len,
            "max_prompt_len": args.max_prompt_len,
            "arrival_rate_rps": args.arrival_rate_rps,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
        "experiments": [],
    }

    for name, url in urls:
        print(f"Running trace benchmark for {name} ...")
        exp = run_trace(url, args.model, prompts, args.max_tokens,
                        args.arrival_rate_rps)
        exp["name"] = name
        output["experiments"].append(exp)

    if len(output["experiments"]) == 2:
        b = output["experiments"][0]
        l = output["experiments"][1]
        output["comparison"] = {
            "ttft_p90_speedup": b["ttft_ms"]["p90"] / l["ttft_ms"]["p90"]
            if l["ttft_ms"]["p90"] else 0.0,
            "ttft_p99_speedup": b["ttft_ms"]["p99"] / l["ttft_ms"]["p99"]
            if l["ttft_ms"]["p99"] else 0.0,
            "e2e_p90_speedup": b["e2e_ms"]["p90"] / l["e2e_ms"]["p90"]
            if l["e2e_ms"]["p90"] else 0.0,
            "e2e_p99_speedup": b["e2e_ms"]["p99"] / l["e2e_ms"]["p99"]
            if l["e2e_ms"]["p99"] else 0.0,
            "slo_attainment_gain": l["slo_attainment"] - b["slo_attainment"],
        }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2)
    print(f"Wrote trace benchmark results to {out}")


if __name__ == "__main__":
    main()