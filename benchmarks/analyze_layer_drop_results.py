#!/usr/bin/env python3
"""
Layer Drop Benchmark Result Analysis Script
Analyzes and compares baseline vs layer drop results across different batch sizes.
Outputs comparison tables to markdown file.
"""

import json
import argparse
from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime


@dataclass
class ExperimentResult:
    """Represents results for a single experiment"""
    batch_size: int
    throughput_rps: float
    ttft_mean: float
    ttft_p50: float
    ttft_p90: float
    ttft_p99: float
    e2e_mean: float
    e2e_p50: float
    e2e_p90: float
    e2e_p99: float
    success_rate: float
    slo_ttft_attainment: float
    slo_e2e_attainment: float


def load_results(file_path: Path) -> Dict[int, ExperimentResult]:
    """Load results from JSON file"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    results = {}
    for exp in data.get('experiments', []):
        bs = exp['batch_size']
        ttft = exp['ttft_ms']
        e2e = exp['e2e_ms']
        
        results[bs] = ExperimentResult(
            batch_size=bs,
            throughput_rps=exp['throughput_rps'],
            ttft_mean=ttft['mean_ms'],
            ttft_p50=ttft['p50_ms'],
            ttft_p90=ttft['p90_ms'],
            ttft_p99=ttft['p99_ms'],
            e2e_mean=e2e['mean_ms'],
            e2e_p50=e2e['p50_ms'],
            e2e_p90=e2e['p90_ms'],
            e2e_p99=e2e['p99_ms'],
            success_rate=(exp['success'] / exp['num_prompts']) * 100,
            slo_ttft_attainment=exp['slo_ttft_attainment_pct'],
            slo_e2e_attainment=exp['slo_e2e_attainment_pct']
        )
    
    return results


def compute_diff(baseline: float, ld: float) -> float:
    """Compute percentage difference (LD vs Baseline)"""
    if baseline == 0:
        return 0.0
    return ((ld - baseline) / baseline) * 100


def format_diff(diff: float, is_higher_better: bool) -> str:
    """Format percentage difference with color-like indicators"""
    if is_higher_better:
        if diff > 0:
            return f"+{diff:.2f}%"
        elif diff < 0:
            return f"{diff:.2f}%"
        else:
            return "0.00%"
    else:
        if diff < 0:
            return f"{diff:.2f}%"
        elif diff > 0:
            return f"+{diff:.2f}%"
        else:
            return "0.00%"


def generate_md_report(baseline_results: Dict[int, ExperimentResult],
                       ld_results: Dict[int, ExperimentResult],
                       output_path: Path) -> None:
    """Generate markdown report with comparison tables"""
    batch_sizes = sorted(set(baseline_results.keys()) | set(ld_results.keys()))
    
    md = []
    md.append("# Layer Drop Benchmark Results Comparison")
    md.append(f"\n*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    md.append("\n## Configuration")
    md.append("| Parameter | Baseline | Layer Drop |")
    md.append("|-----------|----------|------------|")
    md.append("| Drop Ratio | 0.0 | 0.3 |")
    md.append("| TP Size | 2 | 2 |")
    md.append("| Model | Qwen2.5-0.5B-Instruct | Qwen2.5-0.5B-Instruct |")
    
    # Throughput Table
    md.append("\n## Throughput (RPS)")
    md.append("| Batch Size | Baseline | LayerDrop | Diff | Winner |")
    md.append("|------------|----------|-----------|------|--------|")
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        if not base or not ld:
            continue
        diff = compute_diff(base.throughput_rps, ld.throughput_rps)
        winner = "LD" if diff > 0 else "Base" if diff < 0 else "Tie"
        md.append(f"| {bs} | {base.throughput_rps:.2f} | {ld.throughput_rps:.2f} | {format_diff(diff, True)} | {winner} |")
    
    # TTFT p99 Table
    md.append("\n## TTFT p99 (ms)")
    md.append("| Batch Size | Baseline | LayerDrop | Diff | Winner |")
    md.append("|------------|----------|-----------|------|--------|")
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        if not base or not ld:
            continue
        diff = compute_diff(base.ttft_p99, ld.ttft_p99)
        winner = "LD" if diff < 0 else "Base" if diff > 0 else "Tie"
        md.append(f"| {bs} | {base.ttft_p99:.2f} | {ld.ttft_p99:.2f} | {format_diff(diff, False)} | {winner} |")
    
    # E2E p99 Table
    md.append("\n## E2E p99 (ms)")
    md.append("| Batch Size | Baseline | LayerDrop | Diff | Winner |")
    md.append("|------------|----------|-----------|------|--------|")
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        if not base or not ld:
            continue
        diff = compute_diff(base.e2e_p99, ld.e2e_p99)
        winner = "LD" if diff < 0 else "Base" if diff > 0 else "Tie"
        md.append(f"| {bs} | {base.e2e_p99:.2f} | {ld.e2e_p99:.2f} | {format_diff(diff, False)} | {winner} |")
    
    # Summary Statistics
    md.append("\n## Summary Statistics")
    ld_wins = 0
    base_wins = 0
    ties = 0
    
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        if not base or not ld:
            continue
        for b_val, l_val, is_higher in [
            (base.throughput_rps, ld.throughput_rps, True),
            (base.ttft_p99, ld.ttft_p99, False),
            (base.e2e_p99, ld.e2e_p99, False),
        ]:
            diff = compute_diff(b_val, l_val)
            if is_higher:
                if diff > 0:
                    ld_wins += 1
                elif diff < 0:
                    base_wins += 1
                else:
                    ties += 1
            else:
                if diff < 0:
                    ld_wins += 1
                elif diff > 0:
                    base_wins += 1
                else:
                    ties += 1
    
    md.append(f"\n- LayerDrop wins: {ld_wins}")
    md.append(f"- Baseline wins: {base_wins}")
    md.append(f"- Ties: {ties}")
    
    # Best Performing Batch Sizes
    md.append("\n## Best Performing Batch Sizes")
    for name, results, label in [
        ("Baseline", baseline_results, "Base"),
        ("LayerDrop", ld_results, "LD"),
    ]:
        best_throughput = max(results.values(), key=lambda x: x.throughput_rps)
        best_e2e_p99 = min(results.values(), key=lambda x: x.e2e_p99)
        md.append(f"\n### {name}")
        md.append(f"- Best Throughput: batch_size={best_throughput.batch_size}, {best_throughput.throughput_rps:.2f} RPS")
        md.append(f"- Best E2E p99: batch_size={best_e2e_p99.batch_size}, {best_e2e_p99.e2e_p99:.2f} ms")
    
    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"\nMarkdown report written to: {output_path}")


def print_summary(baseline_results: Dict[int, ExperimentResult], 
                  ld_results: Dict[int, ExperimentResult]) -> None:
    """Print comparison summary"""
    
    # Get all batch sizes sorted
    batch_sizes = sorted(set(baseline_results.keys()) | set(ld_results.keys()))
    
    print("=" * 120)
    print(f"{'Layer Drop Benchmark Results Comparison':^120}")
    print(f"{'Baseline: drop_ratio=0.0':<60} {'Layer Drop: drop_ratio=0.3':>60}")
    print("=" * 120)
    
    # Print detailed comparison table
    print(f"\n{'Batch Size':<12} {'Metric':<15} {'Baseline':<15} {'LayerDrop':<15} {'Diff %':<10} {'Winner':<10}")
    print("-" * 80)
    
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        
        if not base or not ld:
            continue
        
        metrics = [
            ("Throughput (RPS)", base.throughput_rps, ld.throughput_rps, True),
            ("TTFT Mean (ms)", base.ttft_mean, ld.ttft_mean, False),
            ("TTFT p50 (ms)", base.ttft_p50, ld.ttft_p50, False),
            ("TTFT p90 (ms)", base.ttft_p90, ld.ttft_p90, False),
            ("TTFT p99 (ms)", base.ttft_p99, ld.ttft_p99, False),
            ("E2E Mean (ms)", base.e2e_mean, ld.e2e_mean, False),
            ("E2E p50 (ms)", base.e2e_p50, ld.e2e_p50, False),
            ("E2E p90 (ms)", base.e2e_p90, ld.e2e_p90, False),
            ("E2E p99 (ms)", base.e2e_p99, ld.e2e_p99, False),
        ]
        
        for i, (name, b_val, l_val, is_higher_better) in enumerate(metrics):
            diff = compute_diff(b_val, l_val)
            
            if is_higher_better:
                winner = "LD" if diff > 0 else "Base" if diff < 0 else "Tie"
            else:
                winner = "LD" if diff < 0 else "Base" if diff > 0 else "Tie"
            
            if i == 0:
                bs_display = str(bs)
            else:
                bs_display = ""
            
            print(f"{bs_display:<12} {name:<15} {b_val:<15.2f} {l_val:<15.2f} {diff:<10.2f} {winner:<10}")
        
        print("-" * 80)
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("-" * 80)
    
    # Count wins
    ld_wins = 0
    base_wins = 0
    ties = 0
    
    for bs in batch_sizes:
        base = baseline_results.get(bs)
        ld = ld_results.get(bs)
        
        if not base or not ld:
            continue
        
        # Compare key metrics
        for b_val, l_val, is_higher in [
            (base.throughput_rps, ld.throughput_rps, True),
            (base.ttft_p99, ld.ttft_p99, False),
            (base.e2e_p99, ld.e2e_p99, False),
        ]:
            diff = compute_diff(b_val, l_val)
            if is_higher:
                if diff > 0:
                    ld_wins += 1
                elif diff < 0:
                    base_wins += 1
                else:
                    ties += 1
            else:
                if diff < 0:
                    ld_wins += 1
                elif diff > 0:
                    base_wins += 1
                else:
                    ties += 1
    
    print(f"LayerDrop wins: {ld_wins}")
    print(f"Baseline wins: {base_wins}")
    print(f"Ties: {ties}")
    
    # Best batch size analysis
    print("\nBest Performing Batch Size Analysis")
    print("-" * 80)
    
    for name, results, label in [
        ("Baseline", baseline_results, "Base"),
        ("LayerDrop", ld_results, "LD"),
    ]:
        best_throughput = max(results.values(), key=lambda x: x.throughput_rps)
        best_e2e_p99 = min(results.values(), key=lambda x: x.e2e_p99)
        
        print(f"{name}:")
        print(f"  Best Throughput: {best_throughput.batch_size} (batch_size) = {best_throughput.throughput_rps:.2f} RPS")
        print(f"  Best E2E p99:    {best_e2e_p99.batch_size} (batch_size) = {best_e2e_p99.e2e_p99:.2f} ms")


def main():
    parser = argparse.ArgumentParser(description='Analyze layer drop benchmark results')
    parser.add_argument('--baseline', type=str, required=True, help='Path to baseline results JSON')
    parser.add_argument('--layer-drop', type=str, required=True, help='Path to layer drop results JSON')
    parser.add_argument('--output', type=str, default='results/layer_drop_comparison.md', 
                        help='Output markdown file path')
    
    args = parser.parse_args()
    
    baseline_path = Path(args.baseline)
    ld_path = Path(args.layer_drop)
    output_path = Path(args.output)
    
    if not baseline_path.exists():
        print(f"Error: Baseline file not found: {baseline_path}")
        return
    
    if not ld_path.exists():
        print(f"Error: Layer drop file not found: {ld_path}")
        return
    
    print(f"Loading baseline results from: {baseline_path}")
    print(f"Loading layer drop results from: {ld_path}")
    
    baseline_results = load_results(baseline_path)
    ld_results = load_results(ld_path)
    
    print(f"\nFound {len(baseline_results)} baseline experiments")
    print(f"Found {len(ld_results)} layer drop experiments")
    
    print_summary(baseline_results, ld_results)
    generate_md_report(baseline_results, ld_results, output_path)


if __name__ == "__main__":
    main()
