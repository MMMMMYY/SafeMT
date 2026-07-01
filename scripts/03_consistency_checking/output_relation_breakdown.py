"""
Multi-dimensional breakdown of output_relation_results.jsonl:
consistency rates at each layer, grouped by MR type / category / prompt_type.

Usage:
    python3 scripts/output_relation_breakdown.py \
        --results data/generated/safemt/sdxl_output_relation_results.jsonl \
        --csv-output data/generated/safemt/sdxl_output_relation_breakdown.csv
"""

import argparse
import csv
import json
from collections import defaultdict


def compute_row(dimension: str, group, items: list[dict]) -> dict:
    n = len(items)
    l1_consistent = sum(1 for r in items if r["layer1_consistent"])
    l2_items = [r for r in items if r["layer2_q16_consistent"] is not None]
    l2_consistent = sum(1 for r in l2_items if r["layer2_q16_consistent"])
    l3_items = [r for r in items if r["layer3_consistent"] is not None]
    l3_consistent = sum(1 for r in l3_items if r["layer3_consistent"])
    final_consistent = sum(1 for r in items if r["final_consistent"])

    flag_l1 = sum(1 for r in items if r["stage"] == "level1_mismatch")
    flag_l2 = sum(1 for r in items if r["stage"] == "level2_mismatch")
    flag_l3 = sum(1 for r in items if r["stage"] == "level3_drift")

    return {
        "dimension": dimension,
        "group": group,
        "n": n,
        "layer1_consistent_pct": round(l1_consistent / n * 100, 1) if n else "",
        "layer2_consistent_pct": round(l2_consistent / len(l2_items) * 100, 1) if l2_items else "",
        "layer3_consistent_pct": round(l3_consistent / len(l3_items) * 100, 1) if l3_items else "",
        "final_consistent_pct": round(final_consistent / n * 100, 1) if n else "",
        "flagged_level1": flag_l1,
        "flagged_level2": flag_l2,
        "flagged_level3": flag_l3,
        "flagged_total": n - final_consistent,
    }


def breakdown_print(rows: list[dict], label: str) -> None:
    print(f"=== 按{label} ===")
    print(f'{"组":20s} {"总数":>6s} {"L1一致%":>9s} {"L2一致%":>9s} {"L3一致%":>9s} {"最终一致%":>9s} '
          f'{"flag_L1":>8s} {"flag_L2":>8s} {"flag_L3":>8s} {"flag合计":>8s}')
    for r in rows:
        print(f"{str(r['group']):20s} {r['n']:6d} "
              f"{r['layer1_consistent_pct']:>8}% {r['layer2_consistent_pct']:>8}% "
              f"{r['layer3_consistent_pct']:>8}% {r['final_consistent_pct']:>8}% "
              f"{r['flagged_level1']:8d} {r['flagged_level2']:8d} {r['flagged_level3']:8d} {r['flagged_total']:8d}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--csv-output", default=None)
    args = parser.parse_args()

    results = [json.loads(l) for l in open(args.results)]
    print(f"Total pairs: {len(results)}\n")

    dimensions = [
        (lambda r: r["mr_id"], "MR类型(8个)", "mr_type"),
        (lambda r: r["category"], "类别(6个)", "category"),
        (lambda r: r["prompt_type"], "prompt_type(3个)", "prompt_type"),
    ]

    all_rows = []
    for key_fn, label, dim_name in dimensions:
        groups = defaultdict(list)
        for r in results:
            groups[key_fn(r)].append(r)
        rows = [compute_row(dim_name, g, groups[g]) for g in sorted(groups, key=str)]
        breakdown_print(rows, label)
        all_rows.extend(rows)

    all_rows.append(compute_row("overall", "ALL", results))
    print("=== Overall ===")
    breakdown_print([all_rows[-1]], "overall")

    if args.csv_output:
        with open(args.csv_output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"Saved CSV -> {args.csv_output}")


if __name__ == "__main__":
    main()
