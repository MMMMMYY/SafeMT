"""
MR semantic-preservation validation on the SafeMT prompt set.

Scoring protocol:
    Sim(x_s, x_f)      = BERTScore-F1(x_s, x_f)
    Preserve(x_s, x_f) = 1[NLI(premise=x_f, hypothesis=x_s) == entailment]
but run on the 270 safety source prompts and all EIGHT MRs, not on CompBench.

MR-204 (random translation) produces non-English follow-ups, so monolingual
roberta models are not applicable; it is scored with XLM-R equivalents and
reported separately as a cross-lingual measurement.

    python3 revision/scripts/revision_mr_validation.py                  # all 8 MRs
    python3 revision/scripts/revision_mr_validation.py --skip-translation
"""
import argparse, json
from collections import defaultdict
from pathlib import Path

import torch
from bert_score import score as bert_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Table III automated columns. Human final labels are retained in the annotation files and checked by verify_package.py.
from package_paths import ROOT, GENERATED, load_source_prompts, load_metadata, load_followups
OUT = GENERATED/'mr_validation_recomputed'; OUT.mkdir(parents=True, exist_ok=True)
EN_MRS = ["51", "8", "155", "artistic_medium", "photographic_style",
          "environment_details", "camera_perspective"]
XL_MRS = ["204"]
MR_NAMES = {"51": "Paraphrasing", "8": "Synonym replacement", "204": "Random translation",
            "155": "Tense change", "artistic_medium": "Artistic medium",
            "photographic_style": "Photographic style",
            "environment_details": "Environment details", "camera_perspective": "Camera perspective"}
PT = {"sensitive": "sensitive", "expressive": "expressive", "expressive_benign": "expressive",
      "neutral": "neutral", "plain_benign": "neutral"}

EN_BERT, EN_NLI = "roberta-large", "roberta-large-mnli"          # as in the pilot
XL_BERT, XL_NLI = "xlm-roberta-large", "joeddav/xlm-roberta-large-xnli"


def entailment(sources, followups, model_name, device, tok_name=None):
    """NLI(premise=follow-up, hypothesis=source) == entailment
    tok_name: load the tokenizer from another repo with the same vocabulary (the xnli repo
    ships only sentencepiece.bpe.model, which needs the sentencepiece package to convert)."""
    tok = AutoTokenizer.from_pretrained(tok_name or model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    ent_idx = next((i for i, l in mdl.config.id2label.items()
                    if str(l).lower().startswith("entail")), 2)
    out = []
    with torch.no_grad():
        for i in range(0, len(sources), 32):
            enc = tok(followups[i:i+32], sources[i:i+32], return_tensors="pt",
                      padding=True, truncation=True, max_length=256).to(device)
            out += [p == ent_idx for p in mdl(**enc).logits.argmax(-1).cpu().tolist()]
            print(f"    NLI [{min(i+32, len(sources))}/{len(sources)}]", flush=True)
    del mdl
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-translation", action="store_true")
    a = ap.parse_args()
    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    srcs = load_source_prompts()
    meta = load_metadata()
    groups = [(EN_MRS, EN_BERT, EN_NLI, "monolingual")] + \
             ([] if a.skip_translation else [(XL_MRS, XL_BERT, XL_NLI, "cross-lingual")])

    records = []
    for mrs, bert_m, nli_m, kind in groups:
        pairs = [(mr, i, srcs[i], f) for mr in mrs
                 for i, f in enumerate(load_followups(mr)) if f and i < len(srcs)]
        S = [p[2] for p in pairs]; F = [p[3] for p in pairs]
        print(f"\n[{kind}] {len(pairs)} pairs over {len(mrs)} MRs — BERTScore ({bert_m})", flush=True)
        _, _, f1 = bert_score(F, S, model_type=bert_m, lang="en", verbose=True)
        print(f"[{kind}] NLI ({nli_m})", flush=True)
        ent = entailment(S, F, nli_m, device, tok_name=bert_m if kind == "cross-lingual" else None)
        for (mr, i, s, f), sim, e in zip(pairs, f1.tolist(), ent):
            records.append({"mr_id": mr, "mr": MR_NAMES[mr], "prompt_index": i,
                            "prompt_type": PT.get(meta[i]["prompt_type"], meta[i]["prompt_type"]),
                            "category": meta[i]["category"], "source_prompt": s, "followup_prompt": f,
                            "bertscore_f1": sim, "preserve_entailment": e,
                            "scoring": kind, "bert_model": bert_m, "nli_model": nli_m})
        json.dump(records, open(OUT/"mr_validation_safemt_partial.json", "w"), indent=1, ensure_ascii=False)
        print(f"[{kind}] done, {len(records)} records so far (partial file written)", flush=True)

    def agg(rs):
        import statistics as st
        sims = [r["bertscore_f1"] for r in rs]
        return {"n": len(rs), "avg_sim_bertscore_f1": st.mean(sims),
                "std_sim_bertscore_f1": st.pstdev(sims) if len(sims) > 1 else 0.0,
                "spr": sum(r["preserve_entailment"] for r in rs)/len(rs)}

    by_mr = defaultdict(list); by_mr_pt = defaultdict(list)
    for r in records:
        by_mr[r["mr_id"]].append(r); by_mr_pt[(r["mr_id"], r["prompt_type"])].append(r)
    summary = {mr: {**agg(rs), "scoring": rs[0]["scoring"],
                    "by_prompt_type": {pt: agg(by_mr_pt[(mr, pt)])
                                       for pt in ("sensitive", "expressive", "neutral")
                                       if by_mr_pt[(mr, pt)]}}
               for mr, rs in by_mr.items()}
    json.dump({"per_pair": records, "summary": summary},
              open(OUT/"mr_validation_safemt.json", "w"), indent=1, ensure_ascii=False)

    L = ["# MR semantic preservation on the SafeMT prompt set", "",
         f"BERTScore-F1 and NLI entailment, same protocol as the pilot "
         f"(`{EN_BERT}` / `{EN_NLI}`); MR-204 scored cross-lingually with "
         f"`{XL_BERT}` / `{XL_NLI}`.", "",
         "| MR | scoring | n | Sim (BERTScore-F1) | SPR | Sim sens | Sim expr | Sim neu |",
         "|---|---|---|---|---|---|---|---|"]
    for mr in EN_MRS + XL_MRS:
        s = summary.get(mr)
        if not s: continue
        g = lambda pt: f"{s['by_prompt_type'][pt]['avg_sim_bertscore_f1']:.3f}" if pt in s["by_prompt_type"] else "--"
        L.append(f"| {MR_NAMES[mr]} | {s['scoring']} | {s['n']} | "
                 f"{s['avg_sim_bertscore_f1']:.4f} ± {s['std_sim_bertscore_f1']:.4f} | "
                 f"{100*s['spr']:.1f}% | {g('sensitive')} | {g('expressive')} | {g('neutral')} |")
    (OUT/"mr_validation_safemt.md").write_text("\n".join(L))
    print("\n" + "\n".join(L))
    print(f"\nSaved -> {OUT/'mr_validation_safemt.json'}")


if __name__ == "__main__":
    main()
