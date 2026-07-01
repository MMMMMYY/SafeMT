"""
Show that CLIP image-text similarity (R-P proxy) cannot distinguish
safety-inconsistent pairs from consistent ones.

Input:  stage2_{sdxl,gpt,gemini}_saliency.jsonl,
        {sdxl,gpt,gemini}_output_relation_results.jsonl
Output: rp_vs_safety.pdf
"""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent.parent  # data/results/stage2_plots → project root

SALIENCY = {
    "sdxl":   ROOT / "data/results/stage2_sdxl_saliency.jsonl",
    "gpt":    ROOT / "data/results/stage2_gpt_saliency.jsonl",
    "gemini": ROOT / "data/results/stage2_gemini_saliency.jsonl",
}
RELATION = {
    "sdxl":   ROOT / "data/generated/safemt/sdxl_output_relation_results.jsonl",
    "gpt":    ROOT / "data/generated/safemt/gpt_image_output_relation_results.jsonl",
    "gemini": ROOT / "data/generated/safemt/gemini_output_relation_results.jsonl",
}

# ── Image-text sim for inconsistent pairs ────────────────────────────────────
inc_img_text = []
for key in ["sdxl", "gpt", "gemini"]:
    for line in open(SALIENCY[key]):
        r = json.loads(line)
        if r.get("source_baseline_sim"):   inc_img_text.append(r["source_baseline_sim"])
        if r.get("followup_baseline_sim"): inc_img_text.append(r["followup_baseline_sim"])

# ── Image-image sim: consistent vs inconsistent ───────────────────────────────
cons_ii, incons_ii = [], []
for key in ["sdxl", "gpt", "gemini"]:
    for line in open(RELATION[key]):
        r = json.loads(line)
        sim = r.get("layer3_clip_similarity")
        if sim is None: continue
        if r.get("final_consistent"):
            cons_ii.append(sim)
        else:
            incons_ii.append(sim)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Left: image-text sim for inconsistent pairs ───────────────────────────────
ax = axes[0]
ax.hist(inc_img_text, bins=40, color="#e15759", edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(np.mean(inc_img_text), color="orange", linewidth=1.5,
           label=f"mean = {np.mean(inc_img_text):.3f}")
ax.axvline(np.median(inc_img_text), color="green", linewidth=1.5, linestyle=":",
           label=f"median = {np.median(inc_img_text):.3f}")
high_rp = sum(1 for x in inc_img_text if x > 0.25)
ax.set_xlabel("CLIP Image-Text Similarity (R-P Proxy)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title(f"Safety-Inconsistent Pairs — Image-Text Alignment\n(SDXL + GPT + Gemini, n={len(inc_img_text)} images)", fontsize=11)
ax.legend(fontsize=9)
ax.text(0.97, 0.95,
    f"sim > 0.25: {high_rp} ({100*high_rp/len(inc_img_text):.1f}%)\n(high alignment = R-P would pass)",
    transform=ax.transAxes, va="top", ha="right", fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

# ── Right: image-image sim, consistent vs inconsistent ───────────────────────
ax = axes[1]
bins = np.linspace(0.5, 1.0, 35)
ax.hist(cons_ii,   bins=bins, color="#4e79a7", edgecolor="white", linewidth=0.5, alpha=0.7,
        label=f"Consistent (n={len(cons_ii)}, μ={np.mean(cons_ii):.3f})")
ax.hist(incons_ii, bins=bins, color="#e15759", edgecolor="white", linewidth=0.5, alpha=0.7,
        label=f"Safety-Inconsistent (n={len(incons_ii)}, μ={np.mean(incons_ii):.3f})")
ax.set_xlabel("CLIP Image-Image Similarity", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("Consistent vs Safety-Inconsistent Pairs\nCLIP Image-Image Similarity", fontsize=11)
ax.legend(fontsize=9)

plt.suptitle("R-P Metric Cannot Distinguish Safety Inconsistency", fontsize=13)
plt.tight_layout()
out = HERE / "rp_vs_safety.pdf"
plt.savefig(out, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")
