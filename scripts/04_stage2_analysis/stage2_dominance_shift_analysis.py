"""
Stage 2 dominance shift analysis — exports full results to CSV.

Classifies each inconsistent pair into:
  anchor_only  — only the anchor word rose in dominance
  context_only — only non-anchor shared words rose
  both         — both anchor and context words rose
  no_shift     — no qualifying dominance shift detected

Usage:
    python3 scripts/stage2_dominance_shift_analysis.py --model sdxl
    python3 scripts/stage2_dominance_shift_analysis.py --model gpt
"""

import argparse
import json
import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent

MIN_RANK_GAIN = 2
MIN_SAL_DELTA = 0.005


def get_suspects(r):
    return [
        s for s in r.get("dominance_gain_suspects", [])
        if s["rank_delta"] <= -MIN_RANK_GAIN and s["saliency_delta"] >= MIN_SAL_DELTA
    ]


def has_dominance_shift(r):
    return len(get_suspects(r)) > 0


def classify_shift(r):
    suspects = get_suspects(r)
    if not suspects:
        return "no_shift"
    anchor = r["anchor_cue"].lower()
    has_anchor  = any(s["word"].lower() == anchor for s in suspects)
    has_context = any(s["word"].lower() != anchor for s in suspects)
    if has_anchor and has_context:
        return "both"
    elif has_anchor:
        return "anchor_only"
    else:
        return "context_only"


def top_suspects(r, n=3):
    hits = get_suspects(r)
    parts = [
        f"{s['word']}({s['source_rank']}→{s['followup_rank']},Δsal={s['saliency_delta']:+.4f})"
        for s in hits[:n]
    ]
    return "; ".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["sdxl", "gpt"], default="sdxl")
    args = parser.parse_args()

    INPUT       = ROOT / f"data/results/stage2_{args.model}_saliency.jsonl"
    OUT_PAIR    = ROOT / f"data/results/stage2_dominance_shift_pairs_{args.model}.csv"
    OUT_WORD    = ROOT / f"data/results/stage2_dominance_shift_words_{args.model}.csv"
    OUT_SUMMARY = ROOT / f"data/results/stage2_dominance_shift_summary_{args.model}.csv"

    records = []
    with open(INPUT) as f:
        for line in f:
            records.append(json.loads(line))

    OUT_PAIR.parent.mkdir(parents=True, exist_ok=True)

    # ── Per-pair table ────────────────────────────────────────────────────────
    pair_fields = [
        "prompt_index", "mr_id", "category", "prompt_type", "anchor_cue",
        "stage", "final_consistent",
        "source_prompt", "followup_prompt",
        "layer3_clip_similarity",
        "source_baseline_sim", "followup_baseline_sim",
        "delta_removed", "delta_added", "shared_word_count",
        "shift_type",           # anchor_only / context_only / both / no_shift
        "has_dominance_shift",
        "num_suspects",
        "num_anchor_suspects",
        "num_context_suspects",
        "top_suspect_words",
        "source_top1_word", "source_top1_saliency",
        "followup_top1_word", "followup_top1_saliency",
    ]

    with open(OUT_PAIR, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=pair_fields)
        w.writeheader()
        for r in records:
            suspects = get_suspects(r)
            anchor = r["anchor_cue"].lower()
            anchor_sus = [s for s in suspects if s["word"].lower() == anchor]
            context_sus = [s for s in suspects if s["word"].lower() != anchor]
            src_sal = r.get("source_saliency", [])
            fup_sal = r.get("followup_saliency", [])
            w.writerow({
                "prompt_index": r["prompt_index"],
                "mr_id": r["mr_id"],
                "category": r["category"],
                "prompt_type": r["prompt_type"],
                "anchor_cue": r["anchor_cue"],
                "stage": r["stage"],
                "final_consistent": r["final_consistent"],
                "source_prompt": r["source_prompt"],
                "followup_prompt": r["followup_prompt"],
                "layer3_clip_similarity": r.get("layer3_clip_similarity", ""),
                "source_baseline_sim": r["source_baseline_sim"],
                "followup_baseline_sim": r["followup_baseline_sim"],
                "delta_removed": "; ".join(r["delta"]["delta_removed"]),
                "delta_added":   "; ".join(r["delta"]["delta_added"]),
                "shared_word_count": len(r["delta"]["shared_words"]),
                "shift_type": classify_shift(r),
                "has_dominance_shift": int(has_dominance_shift(r)),
                "num_suspects": len(suspects),
                "num_anchor_suspects": len(anchor_sus),
                "num_context_suspects": len(context_sus),
                "top_suspect_words": top_suspects(r),
                "source_top1_word": src_sal[0]["word"] if src_sal else "",
                "source_top1_saliency": src_sal[0]["saliency"] if src_sal else "",
                "followup_top1_word": fup_sal[0]["word"] if fup_sal else "",
                "followup_top1_saliency": fup_sal[0]["saliency"] if fup_sal else "",
            })

    print(f"Pair table  → {OUT_PAIR}  ({len(records)} rows)")

    # ── Per-suspect-word table ────────────────────────────────────────────────
    word_fields = [
        "prompt_index", "mr_id", "category", "prompt_type", "anchor_cue",
        "stage", "shift_type",
        "source_prompt", "followup_prompt",
        "suspect_word", "is_anchor",
        "source_rank", "followup_rank", "rank_delta",
        "source_saliency", "followup_saliency", "saliency_delta",
    ]

    word_rows = []
    for r in records:
        suspects = get_suspects(r)
        if not suspects:
            continue
        anchor = r["anchor_cue"].lower()
        stype = classify_shift(r)
        for s in suspects:
            word_rows.append({
                "prompt_index": r["prompt_index"],
                "mr_id": r["mr_id"],
                "category": r["category"],
                "prompt_type": r["prompt_type"],
                "anchor_cue": r["anchor_cue"],
                "stage": r["stage"],
                "shift_type": stype,
                "source_prompt": r["source_prompt"],
                "followup_prompt": r["followup_prompt"],
                "suspect_word": s["word"],
                "is_anchor": int(s["word"].lower() == anchor),
                "source_rank": s["source_rank"],
                "followup_rank": s["followup_rank"],
                "rank_delta": s["rank_delta"],
                "source_saliency": s["source_saliency"],
                "followup_saliency": s["followup_saliency"],
                "saliency_delta": s["saliency_delta"],
            })

    with open(OUT_WORD, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=word_fields)
        w.writeheader()
        w.writerows(word_rows)

    print(f"Word table  → {OUT_WORD}  ({len(word_rows)} rows)")

    # ── Summary table ─────────────────────────────────────────────────────────
    SHIFT_TYPES = ["anchor_only", "context_only", "both", "no_shift"]

    def make_summary(dimension):
        keys = sorted(set(r[dimension] for r in records))
        rows = []
        for k in keys:
            subset = [r for r in records if r[dimension] == k]
            n = len(subset)
            type_counts = Counter(classify_shift(r) for r in subset)
            rows.append({
                "dimension": dimension,
                "value": k,
                "total": n,
                "anchor_only":   type_counts["anchor_only"],
                "context_only":  type_counts["context_only"],
                "both":          type_counts["both"],
                "no_shift":      type_counts["no_shift"],
                "any_shift":     n - type_counts["no_shift"],
                "anchor_shift":  type_counts["anchor_only"] + type_counts["both"],
                "context_shift": type_counts["context_only"] + type_counts["both"],
                "anchor_only_pct":   round(100 * type_counts["anchor_only"] / n, 1),
                "context_only_pct":  round(100 * type_counts["context_only"] / n, 1),
                "both_pct":          round(100 * type_counts["both"] / n, 1),
                "no_shift_pct":      round(100 * type_counts["no_shift"] / n, 1),
                "any_shift_pct":     round(100 * (n - type_counts["no_shift"]) / n, 1),
                "anchor_shift_pct":  round(100 * (type_counts["anchor_only"] + type_counts["both"]) / n, 1),
                "context_shift_pct": round(100 * (type_counts["context_only"] + type_counts["both"]) / n, 1),
            })
        return rows

    summary_rows = []

    # Overall
    n = len(records)
    tc = Counter(classify_shift(r) for r in records)
    summary_rows.append({
        "dimension": "overall", "value": "all", "total": n,
        "anchor_only": tc["anchor_only"], "context_only": tc["context_only"],
        "both": tc["both"], "no_shift": tc["no_shift"],
        "any_shift": n - tc["no_shift"],
        "anchor_shift": tc["anchor_only"] + tc["both"],
        "context_shift": tc["context_only"] + tc["both"],
        "anchor_only_pct":   round(100 * tc["anchor_only"] / n, 1),
        "context_only_pct":  round(100 * tc["context_only"] / n, 1),
        "both_pct":          round(100 * tc["both"] / n, 1),
        "no_shift_pct":      round(100 * tc["no_shift"] / n, 1),
        "any_shift_pct":     round(100 * (n - tc["no_shift"]) / n, 1),
        "anchor_shift_pct":  round(100 * (tc["anchor_only"] + tc["both"]) / n, 1),
        "context_shift_pct": round(100 * (tc["context_only"] + tc["both"]) / n, 1),
    })

    for dim in ("mr_id", "category", "prompt_type", "stage"):
        summary_rows.extend(make_summary(dim))

    summary_fields = [
        "dimension", "value", "total",
        "anchor_only", "context_only", "both", "no_shift", "any_shift",
        "anchor_shift", "context_shift",
        "anchor_only_pct", "context_only_pct", "both_pct", "no_shift_pct",
        "any_shift_pct", "anchor_shift_pct", "context_shift_pct",
    ]
    with open(OUT_SUMMARY, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=summary_fields)
        w.writeheader()
        w.writerows(summary_rows)

    print(f"Summary     → {OUT_SUMMARY}  ({len(summary_rows)} rows)")
    print()

    # Console print
    print(f"{'dimension':15s}  {'value':25s}  {'total':>5}  "
          f"{'anchor_only':>12}  {'context_only':>13}  {'both':>5}  {'no_shift':>9}  "
          f"{'anchor%':>8}  {'context%':>9}")
    print("-" * 110)
    for row in summary_rows:
        print(f"{row['dimension']:15s}  {str(row['value']):25s}  {row['total']:5d}  "
              f"{row['anchor_only']:5d}({row['anchor_only_pct']:4.1f}%)  "
              f"{row['context_only']:5d}({row['context_only_pct']:4.1f}%)  "
              f"{row['both']:3d}({row['both_pct']:3.1f}%)  "
              f"{row['no_shift']:5d}({row['no_shift_pct']:4.1f}%)  "
              f"{row['anchor_shift_pct']:6.1f}%  "
              f"{row['context_shift_pct']:7.1f}%")


if __name__ == "__main__":
    main()
