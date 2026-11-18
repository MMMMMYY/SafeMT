#!/usr/bin/env python3
"""
Figure 5: compare Stage-2 saliency patterns between pairs flagged in Stage 1 and pairs
that were not (matched consistent-pair control).

Input : data/results/stage2_{sdxl,gpt,gemini}_saliency_allpairs.jsonl  (stage2_saliency_attribution.py --all-pairs)
Output: revision/stage2_control.json, revision/stage2_control.md

RQ2 metrics:
  anchor_shift   = rank_f(anchor)/N_f - rank_s(anchor)/N_s        (<0: anchor more salient in follow-up)
  delta_top_rel  = min_{w in delta_added} rank_f(w) / N_f          (<0.5: MR-introduced component in the more salient half)
Populations: flagged = level2_mismatch + level3_drift (level1 refusal mismatches have one image only and are
excluded, as in the paper); unflagged = level3_consistent.
Tests: two-sided permutation test on the difference in proportion (delta_top_rel<0.5) and on the difference in
mean anchor_shift, 10,000 permutations; plus a 95% bootstrap CI on the proportion difference.
"""
import json, random, re, sys, statistics as st
from pathlib import Path
from difflib import SequenceMatcher

from package_paths import ROOT, REV, STAGE2, GENERATED
# Anchor alignment for the paired saliency metrics.
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


# --- end copy

random.seed(0)
MODELS = {"gpt": "GPT-Image", "gemini": "Gemini-Image", "sdxl": "SDXL"}
FLAGGED = {"level2_mismatch", "level3_drift"}
UNFLAGGED = {"level3_consistent"}

def metrics(r):
    if "single_side" in r:          # refusal-mismatch record (one image), handled separately
        return None
    anchor = r["anchor_cue"].lower()
    src_sal, fup_sal = r.get("source_saliency", []), r.get("followup_saliency", [])
    delta_added = r["delta"]["delta_added"]
    if not delta_added or not src_sal or not fup_sal:
        return None
    src_words = [s["word"].lower() for s in src_sal]
    fup_words = [s["word"].lower() for s in fup_sal]
    eff = anchor if anchor in fup_words else None
    if eff is None:
        eq = find_equiv_anchor(r["source_prompt"], r["followup_prompt"], anchor)
        if eq and eq in fup_words:
            eff = eq
    if eff is None or anchor not in src_words:
        return None
    src_rank = next(s["rank"] for s in src_sal if s["word"].lower() == anchor)
    fup_rank = next(s["rank"] for s in fup_sal if s["word"].lower() == eff)
    anchor_shift = fup_rank / len(fup_sal) - src_rank / len(src_sal)
    fmap = {s["word"].lower(): s["rank"] for s in fup_sal}
    da = [fmap[w] for w in delta_added if w in fmap]
    if not da:
        return None
    return {"anchor_shift": anchor_shift, "delta_top_rel": min(da) / len(fup_sal),
            "stage": r["stage"], "mr_id": str(r["mr_id"]), "prompt_type": r["prompt_type"]}

def summ(rows):
    n = len(rows)
    if n == 0:
        return {"n": 0}
    a = [x["anchor_shift"] for x in rows]; d = [x["delta_top_rel"] for x in rows]
    return {"n": n,
            "mr_component_salient_half_pct": round(100 * sum(1 for v in d if v < 0.5) / n, 1),
            "anchor_more_salient_pct": round(100 * sum(1 for v in a if v < 0) / n, 1),
            "anchor_less_salient_pct": round(100 * sum(1 for v in a if v > 0) / n, 1),
            "anchor_shift_mean": round(st.mean(a), 4), "anchor_shift_median": round(st.median(a), 4),
            "delta_top_rel_mean": round(st.mean(d), 4)}

def perm_test(x, y, stat, n_perm=10000):
    obs = stat(x) - stat(y); pool = x + y; nx = len(x); cnt = 0
    for _ in range(n_perm):
        random.shuffle(pool)
        if abs(stat(pool[:nx]) - stat(pool[nx:])) >= abs(obs):
            cnt += 1
    return obs, (cnt + 1) / (n_perm + 1)

def boot_ci(x, y, stat, n_boot=2000):
    diffs = []
    for _ in range(n_boot):
        bx = [random.choice(x) for _ in x]; by = [random.choice(y) for _ in y]
        diffs.append(stat(bx) - stat(by))
    diffs.sort()
    return diffs[int(0.025 * n_boot)], diffs[int(0.975 * n_boot)]

prop = lambda v: sum(1 for t in v if t < 0.5) / len(v)
mean = lambda v: sum(v) / len(v)

def main():
    # Figure 5 and Section V-C: compare flagged and unflagged paired images.
    # Figure 4 reuses metrics() and the single-side records, without rerunning these tests.
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=GENERATED/'stage2_recomputed')
    output_dir = ap.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(0)
    out = {}; md = ["# Stage 2: flagged vs. unflagged pairs\n",
                    "Same metrics as the submitted RQ2 (anchor_shift, delta_top_rel); flagged = label mismatch + semantic drift; "
                    "unflagged = pairs passing all three checks. Refusal mismatches excluded (single image).\n"]
    for tag, name in MODELS.items():
        path = STAGE2 / f"stage2_{tag}_saliency_allpairs.jsonl"
        raw = [json.loads(l) for l in open(path)]
        rows = [m for m in (metrics(r) for r in raw) if m]
        # level-1 (refusal mismatch): generated-side anchor normalized rank, top-30% share, per direction
        l1 = [r for r in raw if "single_side" in r and r.get("anchor_rank_rel") is not None]
        level1 = {}
        for side in ("source", "followup"):
            v = [r["anchor_rank_rel"] for r in l1 if r["single_side"] == side]
            if v:
                level1[f"generated_side={side}"] = {"n": len(v), "anchor_top30_pct": round(100 * sum(1 for x in v if x <= 0.30) / len(v), 1),
                                                   "anchor_rank_rel_mean": round(st.mean(v), 3)}
        fl = [r for r in rows if r["stage"] in FLAGGED]; un = [r for r in rows if r["stage"] in UNFLAGGED]
        res = {"level1_refusal_mismatch": level1, "flagged": summ(fl), "unflagged": summ(un),
               "flagged_by_type": {s: summ([r for r in fl if r["stage"] == s]) for s in sorted(FLAGGED)}}
        if fl and un:
            dfl, dun = [r["delta_top_rel"] for r in fl], [r["delta_top_rel"] for r in un]
            afl, aun = [r["anchor_shift"] for r in fl], [r["anchor_shift"] for r in un]
            o, p = perm_test(dfl, dun, prop); lo, hi = boot_ci(dfl, dun, prop)
            res["test_mr_component_salient_half"] = {"diff_pp": round(100 * o, 1), "ci95_pp": [round(100 * lo, 1), round(100 * hi, 1)], "p_perm": round(p, 4)}
            o2, p2 = perm_test(afl, aun, mean)
            res["test_anchor_shift_mean"] = {"diff": round(o2, 4), "p_perm": round(p2, 4)}
            # by MR, descriptive
            res["by_mr"] = {}
            for mr in sorted({r["mr_id"] for r in rows}):
                f_ = [r for r in fl if r["mr_id"] == mr]; u_ = [r for r in un if r["mr_id"] == mr]
                res["by_mr"][mr] = {"flagged": summ(f_), "unflagged": summ(u_)}
        out[tag] = res
        f, u = res["flagged"], res["unflagged"]
        md.append(f"\n## {name}\n")
        for k, v in level1.items():
            md.append(f"refusal mismatch, {k}: n={v['n']}, anchor in top-30% = {v['anchor_top30_pct']}%  ")
        md.append("| population | n | MR-introduced in more salient half | anchor more salient | anchor less salient | mean anchor_shift |")
        md.append("|---|---|---|---|---|---|")
        for lab, s in [("flagged (label+drift)", f), ("unflagged", u)] + [(k, v) for k, v in res["flagged_by_type"].items()]:
            if s["n"]:
                md.append(f"| {lab} | {s['n']} | {s['mr_component_salient_half_pct']}% | {s['anchor_more_salient_pct']}% | {s['anchor_less_salient_pct']}% | {s['anchor_shift_mean']:+.3f} |")
        if "test_mr_component_salient_half" in res:
            t = res["test_mr_component_salient_half"]; t2 = res["test_anchor_shift_mean"]
            md.append(f"\nMR-introduced-in-salient-half: flagged − unflagged = {t['diff_pp']:+.1f} pp, 95% CI [{t['ci95_pp'][0]:+.1f}, {t['ci95_pp'][1]:+.1f}], permutation p = {t['p_perm']}  ")
            md.append(f"mean anchor_shift: flagged − unflagged = {t2['diff']:+.3f}, permutation p = {t2['p_perm']}")

    (output_dir / "stage2_control.json").write_text(json.dumps(out, indent=1))
    (output_dir / "stage2_control.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

if __name__ == '__main__':
    main()
