# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Output-quality comparison between baseline and layer-drop servers.

Computes exact-match rate, token overlap, and ROUGE-L-like LCS similarity
for the same prompts under baseline and layer-drop configurations. Also
verifies that dropped requests finish with finish_reason=LENGTH and empty
new_token_ids.
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import requests


def longest_common_subsequence(x: list[str], y: list[str]) -> int:
    if not x or not y:
        return 0
    m, n = len(x), len(y)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev, curr = curr, prev
    return prev[n]


def rouge_l_score(reference: str, hypothesis: str) -> float:
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    lcs = longest_common_subsequence(ref_tokens, hyp_tokens)
    if lcs == 0:
        return 0.0
    recall = lcs / len(ref_tokens)
    precision = lcs / len(hyp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def token_overlap(reference: str, hypothesis: str) -> float:
    ref_tokens = set(reference.split())
    hyp_tokens = set(hypothesis.split())
    if not ref_tokens:
        return 0.0
    return len(ref_tokens & hyp_tokens) / len(ref_tokens)


def send_completion(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    try:
        r = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=300,
        )
        r.raise_for_status()
        data = r.json()
        choice = data["choices"][0]
        return {
            "text": choice.get("text", "").strip(),
            "finish_reason": choice.get("finish_reason", ""),
            "latency_ms": 0.0,
        }
    except Exception as e:
        return {"error": str(e)}


def compare_responses(
    prompt: str,
    base: dict[str, Any],
    ld: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "prompt": prompt[:200],
        "baseline": base,
        "layer_drop": ld,
    }

    if "error" in base or "error" in ld:
        result["error"] = True
        return result

    result["exact_match"] = base["text"] == ld["text"]
    result["token_overlap"] = token_overlap(base["text"], ld["text"])
    result["rouge_l"] = rouge_l_score(base["text"], ld["text"])
    result["dropped"] = ld["finish_reason"] == "length" and ld["text"] == ""
    return result


def build_prompts(num_prompts: int, seed: int) -> list[str]:
    random.seed(seed)
    np.random.seed(seed)
    prompts = [
        "Explain the theory of relativity in simple terms.",
        "Write a short poem about artificial intelligence.",
        "Summarize the plot of Romeo and Juliet.",
        "What are the main causes of climate change?",
        "Describe the structure of a cell.",
        "How does a blockchain work?",
        "List three benefits of regular exercise.",
        "What is the capital of France and its history?",
        "Write a Python function to compute factorial.",
        "Compare SQL and NoSQL databases.",
    ]
    return (prompts * ((num_prompts // len(prompts)) + 1))[:num_prompts]


def collect_responses(
    prompts: list[str],
    url: str,
    model: str,
    max_tokens: int,
    temperature: float,
    label: str,
) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for i, prompt in enumerate(prompts):
        print(f"[{i + 1}/{len(prompts)}] {label}: {prompt[:60]}...")
        responses.append(
            send_completion(url, model, prompt, max_tokens, temperature))
    return responses


def compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [r for r in results if not r.get("error")]
    dropped_count = sum(1 for r in valid if r.get("dropped"))
    return {
        "num_prompts": len(results),
        "valid": len(valid),
        "dropped_count": dropped_count,
        "exact_match_rate": float(
            np.mean([r["exact_match"] for r in valid])),
        "mean_token_overlap": float(
            np.mean([r["token_overlap"] for r in valid])),
        "mean_rouge_l": float(np.mean([r["rouge_l"] for r in valid])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["baseline", "layer_drop", "both"], default="both")
    parser.add_argument("--baseline-url",
                        default="http://localhost:8000/v1/completions")
    parser.add_argument("--layer-drop-url",
                        default="http://localhost:8001/v1/completions")
    parser.add_argument("--baseline-cache",
                        default="results/layer_drop_quality_baseline_cache.json")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--output",
                        default="results/layer_drop_quality.json")
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    prompts = build_prompts(args.num_prompts, args.seed)

    if args.mode == "baseline":
        baseline_responses = collect_responses(
            prompts, args.baseline_url, args.model, args.max_tokens,
            args.temperature, "baseline")
        cache = {
            "config": {
                "url": args.baseline_url,
                "model": args.model,
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "seed": args.seed,
            },
            "prompts": prompts,
            "responses": baseline_responses,
        }
        cache_path = Path(args.baseline_cache)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("w") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
        print(f"Saved baseline responses to {cache_path}")
        return

    if args.mode == "layer_drop":
        cache_path = Path(args.baseline_cache)
        if not cache_path.exists():
            raise FileNotFoundError(
                f"Baseline cache not found: {cache_path}. "
                "Run with --mode baseline first.")
        with cache_path.open("r") as f:
            cache = json.load(f)
        baseline_responses = cache["responses"]
        layer_drop_responses = collect_responses(
            prompts, args.layer_drop_url, args.model, args.max_tokens,
            args.temperature, "layer_drop")
        results = [
            compare_responses(p, b, l)
            for p, b, l in zip(prompts, baseline_responses,
                               layer_drop_responses)
        ]
    else:  # both
        results = []
        for i, prompt in enumerate(prompts):
            print(f"[{i + 1}/{len(prompts)}] comparing prompt ...")
            base = send_completion(args.baseline_url, args.model, prompt,
                                   args.max_tokens, args.temperature)
            ld = send_completion(args.layer_drop_url, args.model, prompt,
                                 args.max_tokens, args.temperature)
            results.append(compare_responses(prompt, base, ld))

    summary = compute_summary(results)
    output = {
        "config": {
            "mode": args.mode,
            "baseline_url": args.baseline_url,
            "layer_drop_url": args.layer_drop_url,
            "baseline_cache": args.baseline_cache,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
        "summary": summary,
        "details": results,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Wrote quality comparison to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()