"""
Compute scatter data (anchor_shift + delta_top_rel) for GPT and SDXL.
For cases where anchor word is replaced (MR-8 synonym / MR-204 translation),
use SequenceMatcher alignment to find the equivalent word in followup prompt.
"""
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

STOPWORDS = {
    "a","an","the","in","on","at","with","and","or","of","to","for",
    "by","from","is","are","was","were","be","this","that","their",
    "its","as","who","which","over","under","into","through","near",
    "beside","between","another","other","some","any","each","up","during",
}

def get_words(prompt):
    return re.findall(r"\b\w+\b", prompt)

def find_equiv_anchor(src_prompt, fup_prompt, anchor):
    """
    Use SequenceMatcher to find which word in followup
    corresponds to the anchor position in source.
    Returns the equivalent word (lowercase) or None.
    """
    src_words = get_words(src_prompt)
    fup_words = get_words(fup_prompt)

    # Find anchor position in source
    anchor_positions = [i for i, w in enumerate(src_words) if w.lower() == anchor.lower()]
    if not anchor_positions:
        return None
    anchor_pos = anchor_positions[0]

    matcher = SequenceMatcher(None, [w.lower() for w in src_words],
                                    [w.lower() for w in fup_words])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace" and i1 <= anchor_pos < i2:
            # anchor was replaced by words in fup[j1:j2]
            replaced = fup_words[j1:j2]
            # pick the non-stopword one
            candidates = [w.lower() for w in replaced if w.lower() not in STOPWORDS]
            return candidates[0] if candidates else None
        elif tag == "equal" and i1 <= anchor_pos < i2:
            # anchor unchanged (shouldn't happen if we're here, but just in case)
            offset = anchor_pos - i1
            return fup_words[j1 + offset].lower()
    return None


def compute_scatter(saliency_path, out_path, model_name):
    records = [json.loads(l) for l in open(saliency_path)]
    out = []
    skipped = 0
    equiv_used = 0

    for r in records:
        anchor = r["anchor_cue"].lower()
        src_sal = r.get("source_saliency", [])
        fup_sal = r.get("followup_saliency", [])
        delta_added = r["delta"]["delta_added"]

        if not delta_added:
            skipped += 1
            continue

        src_words = [s["word"].lower() for s in src_sal]
        fup_words = [s["word"].lower() for s in fup_sal]

        # Try direct match first
        effective_anchor = anchor if anchor in fup_words else None

        # If not found, try to find equivalent via alignment
        if effective_anchor is None:
            equiv = find_equiv_anchor(r["source_prompt"], r["followup_prompt"], anchor)
            if equiv and equiv in fup_words:
                effective_anchor = equiv
                equiv_used += 1

        if effective_anchor is None or anchor not in src_words:
            skipped += 1
            continue

        src_rank = next(s["rank"] for s in src_sal if s["word"].lower() == anchor)
        fup_rank = next(s["rank"] for s in fup_sal if s["word"].lower() == effective_anchor)
        anchor_shift = (fup_rank / len(fup_sal)) - (src_rank / len(src_sal))

        fup_rank_map = {s["word"].lower(): s["rank"] for s in fup_sal}
        da_ranks = [fup_rank_map[w] for w in delta_added if w in fup_rank_map]
        if not da_ranks:
            skipped += 1
            continue
        delta_top_rel = min(da_ranks) / len(fup_sal)

        out.append({
            "anchor_shift": anchor_shift,
            "delta_top_rel": delta_top_rel,
            "mr_id": str(r["mr_id"]),
            "category": r["category"],
            "prompt_type": r["prompt_type"],
            "stage": r["stage"],
            "anchor": anchor,
            "effective_anchor": effective_anchor,
        })

    print(f"[{model_name}] Valid: {len(out)}, Skipped: {skipped}, Equiv anchor used: {equiv_used}")
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"  Saved: {out_path}")
    return out


ROOT = Path("/Users/mamengyao/Documents/UNSW/T2I")

compute_scatter(
    ROOT / "data/results/stage2_gpt_saliency.jsonl",
    "/tmp/scatter_gpt_v2.json",
    "GPT"
)

compute_scatter(
    ROOT / "data/results/stage2_sdxl_saliency.jsonl",
    "/tmp/scatter_sdxl_v2.json",
    "SDXL"
)

compute_scatter(
    ROOT / "data/results/stage2_gemini_saliency.jsonl",
    ROOT / "data/results/stage2_plots/scatter_gemini_v2.json",
    "Gemini"
)
