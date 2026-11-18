"""
SafeMT Final Report Generator

Reads output relation results and source prompts, then prints a structured
summary report matching the SafeMT evaluation framework output format.

Usage:
    python3 safemt_report.py --model sdxl
    python3 safemt_report.py --model gpt
    python3 safemt_report.py --model gemini
    python3 safemt_report.py --model all
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RESULT_FILES = {
    "sdxl":   ROOT / "dataset/consistency_results/sdxl_output_relation_results.jsonl",
    "gpt":    ROOT / "dataset/consistency_results/gpt_image_output_relation_results.jsonl",
    "gemini": ROOT / "dataset/consistency_results/gemini_output_relation_results.jsonl",
}

MODEL_LABELS = {
    "sdxl":   "Stable Diffusion XL Base 1.0",
    "gpt":    "GPT-Image-1 (OpenAI)",
    "gemini": "Gemini 2.5 Flash Image (Vertex AI)",
}

MR_LABELS = {
    "51":                  "MR-51  Paraphrasing",
    "8":                   "MR-8   Synonym Replacement",
    "204":                 "MR-204 Translation",
    "155":                 "MR-155 Tense Change",
    "artistic_medium":     "MR     Artistic Medium",
    "photographic_style":  "MR     Photographic Style",
    "environment_details": "MR     Environment Details",
    "camera_perspective":  "MR     Camera Perspective",
}

SOURCE_PROMPTS = ROOT / "dataset/source_prompts/anchored_prompt_groups_v2.jsonl"


def load_results(model_key: str):
    path = RESULT_FILES[model_key]
    return [json.loads(line) for line in path.open()]


def load_source_prompts(n_preview=3):
    groups = [json.loads(l) for l in SOURCE_PROMPTS.open()]
    prompts = []
    for g in groups:
        for pair in g.get("pairs", []):
            prompts.append(pair["sensitive"])
    return prompts


def compute_stats(records):
    total = len(records)
    l1_incons = sum(1 for r in records if not r["layer1_consistent"])
    # Layer 2 is evaluated only when both sides generate images.
    # Both-refused pairs have a null Layer-2 result and remain consistent.
    l2_incons = sum(1 for r in records if r.get("stage") == "level2_mismatch")
    l3_incons = sum(1 for r in records if r.get("stage") == "level3_drift")
    flagged   = l1_incons + l2_incons + l3_incons
    safe_rate = (total - flagged) / total * 100

    by_mr = defaultdict(lambda: {"total": 0, "inconsistent": 0})
    for r in records:
        mr = r["mr_id"]
        by_mr[mr]["total"] += 1
        if not r["final_consistent"]:
            by_mr[mr]["inconsistent"] += 1

    by_category = defaultdict(lambda: {"total": 0, "inconsistent": 0})
    for r in records:
        cat = r["category"]
        by_category[cat]["total"] += 1
        if not r["final_consistent"]:
            by_category[cat]["inconsistent"] += 1

    return {
        "total":      total,
        "flagged":    flagged,
        "safe_rate":  safe_rate,
        "l1":         l1_incons,
        "l2":         l2_incons,
        "l3":         l3_incons,
        "by_mr":      dict(by_mr),
        "by_category": dict(by_category),
    }


SCROLL_TOP    = "  ╔══════════════════════════════════════════╗"
SCROLL_BOTTOM = "  ╚══════════════════════════════════════════╝"
SCROLL_LINE   = "  ║  {:<40}║"
SCROLL_SEP    = "  ║  {:<40}║".format("─" * 40)


def row(text=""):
    return SCROLL_LINE.format(text)


def print_report(model_key: str):
    records = load_results(model_key)
    prompts = load_source_prompts()
    stats   = compute_stats(records)

    n_sources = len(set(r["prompt_index"] for r in records))

    # pick 2 preview prompts + ellipsis
    preview = prompts[:2]

    lines = []
    lines.append(SCROLL_TOP)
    lines.append(row())
    lines.append(row(f"Tested Model:"))
    lines.append(row(f"  {MODEL_LABELS[model_key]}"))
    lines.append(row())
    lines.append(row("Source Inputs:"))
    for p in preview:
        short = (p[:36] + "...") if len(p) > 36 else p
        lines.append(row(f"  {short}"))
    lines.append(row(f"  ......"))
    lines.append(row(f"  ({n_sources} prompts × 8 MRs = {stats['total']} pairs)"))
    lines.append(row())
    lines.append(SCROLL_SEP)
    lines.append(row())
    lines.append(row("Consistency:"))
    lines.append(row(f"  rate: {stats['safe_rate']:.1f}%  "
                     f"({stats['total'] - stats['flagged']}/{stats['total']} consistent)"))
    lines.append(row())
    lines.append(row("Flagged Inconsistent Pairs:"))
    lines.append(row(f"  Total : {stats['flagged']}"))
    lines.append(row(f"  L1 (block/gen mismatch) : {stats['l1']}"))
    lines.append(row(f"  L2 (classifier mismatch): {stats['l2']}"))
    lines.append(row(f"  L3 (semantic drift)      : {stats['l3']}"))
    lines.append(row())
    lines.append(SCROLL_SEP)
    lines.append(row())
    lines.append(row("Component Inconsistency (by MR):"))
    for mr_id, label in MR_LABELS.items():
        d = stats["by_mr"].get(mr_id, {"total": 0, "inconsistent": 0})
        pct = d["inconsistent"] / d["total"] * 100 if d["total"] else 0
        entry = f"  {label}: {d['inconsistent']}/{d['total']} ({pct:.1f}%)"
        if len(entry) > 40:
            entry = entry[:39] + "…"
        lines.append(row(entry))
    lines.append(row())
    lines.append(SCROLL_SEP)
    lines.append(row())
    lines.append(row("Component Inconsistency (by Category):"))
    for cat, d in sorted(stats["by_category"].items()):
        pct = d["inconsistent"] / d["total"] * 100 if d["total"] else 0
        short_cat = cat.replace("_", " ")
        lines.append(row(f"  {short_cat}: {d['inconsistent']}/{d['total']} ({pct:.1f}%)"))
    lines.append(row())
    lines.append(SCROLL_BOTTOM)
    lines.append(f"{'Final Report — SafeMT':^46}")

    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Print SafeMT final report.")
    parser.add_argument("--model", choices=["sdxl", "gpt", "gemini", "all"],
                        default="all", help="Model to report on")
    parser.add_argument("--output", default=None,
                        help="Save report to this file (default: safemt_report.txt next to this script)")
    args = parser.parse_args()

    models = ["sdxl", "gpt", "gemini"] if args.model == "all" else [args.model]

    out_path = Path(args.output) if args.output else Path(__file__).parent / "safemt_report.txt"

    import io, sys
    buf = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buf

    for i, m in enumerate(models):
        if i > 0:
            print("\n")
        print_report(m)

    sys.stdout = old_stdout
    content = buf.getvalue()

    print(content)
    out_path.write_text(content, encoding="utf-8")
    print(f"\n[Saved → {out_path}]")


if __name__ == "__main__":
    main()
