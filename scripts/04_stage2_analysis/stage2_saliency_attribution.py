"""
Stage 2: CLIP saliency attribution analysis.

For each inconsistent (source, followup) pair from output_relation results:
  1. Compute per-word CLIP saliency for source image × source prompt
  2. Compute per-word CLIP saliency for followup image × followup prompt
  3. Locate Δ (words that changed between source and followup prompts)
  4. Find non-Δ words whose saliency rank increased significantly
     → these are "indirect causation" candidates

Usage:
    python3 scripts/stage2_saliency_attribution.py \
        --model sdxl \
        --output data/results/stage2_sdxl_saliency.jsonl

    # Only inconsistent pairs (default):
    python3 scripts/stage2_saliency_attribution.py --model sdxl

    # All pairs (for full dataset coverage):
    python3 scripts/stage2_saliency_attribution.py --model sdxl --all-pairs
"""

import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STOPWORDS = {
    "a","an","the","in","on","at","with","and","or","of","to","for",
    "by","from","is","are","was","were","be","this","that","their",
    "its","as","who","which","over","under","into","through","near",
    "beside","between","another","other","some","any","each","up","during",
}

# ── manifest paths per model tag ─────────────────────────────────────────────

MODEL_CONFIGS = {
    "sdxl": {
        "relation_file": ROOT / "data/generated/safemt/sdxl_output_relation_results.jsonl",
        "source_manifest": ROOT / "data/generated/safemt/output/manifest_0_1350.jsonl",
        "followup_manifest": [
            ROOT / "data/generated/safemt/output/manifest_0_1350.jsonl",
            ROOT / "data/generated/safemt/output_modifier/manifest_0_1080.jsonl",
        ],
        "image_root": ROOT / "data/generated/safemt",
    },
    "gpt": {
        "relation_file": ROOT / "data/generated/safemt/gpt_image_output_relation_results.jsonl",
        "source_manifest": ROOT / "data/generated/safemt/openai_gpt-image-1/manifest.jsonl",
        "followup_manifest": [
            ROOT / "data/generated/safemt/openai_gpt-image-1/manifest.jsonl",
        ],
        "image_root": ROOT / "data/generated/safemt",
    },
    "gemini": {
        "relation_file": ROOT / "data/generated/safemt/gemini_output_relation_results.jsonl",
        "source_manifest": ROOT / "data/generated/safemt/gemini-vertex_gemini-2_5-flash-image/manifest.jsonl",
        "followup_manifest": [
            ROOT / "data/generated/safemt/gemini-vertex_gemini-2_5-flash-image/manifest.jsonl",
        ],
        "image_root": ROOT / "data/generated/safemt",
    },
}


# ── CLIP helpers (reuse clip_image_saliency logic) ───────────────────────────

def load_clip():
    import torch
    import clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess, device


def encode_image(model, preprocess, device, image_path: Path) -> np.ndarray:
    import torch
    from PIL import Image
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def encode_text(model, device, text: str) -> np.ndarray:
    import torch
    import clip
    with torch.no_grad():
        tok = clip.tokenize([text], truncate=True).to(device)
        emb = model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]


def cosine_sim(a, b) -> float:
    return float(np.dot(a, b))


def ablate_word(prompt: str, word: str, occurrence: int = 0) -> str:
    count = 0
    def replacer(m):
        nonlocal count
        if m.group(0).lower() == word.lower():
            if count == occurrence:
                count += 1
                return ""
            count += 1
        return m.group(0)
    result = re.sub(r"\b\w+\b", replacer, prompt)
    return " ".join(result.split())


def compute_saliency(image_path: Path, prompt: str, model, preprocess, device) -> tuple[list[dict], float]:
    """Returns (word_saliency_list, baseline_sim). List sorted by saliency desc."""
    img_emb = encode_image(model, preprocess, device, image_path)
    txt_emb = encode_text(model, device, prompt)
    baseline = cosine_sim(img_emb, txt_emb)

    words = re.findall(r"\b\w+\b", prompt)
    seen: dict[str, int] = {}
    results = []

    for pos, word in enumerate(words):
        wl = word.lower()
        if wl in STOPWORDS:
            continue
        occ = seen.get(wl, 0)
        seen[wl] = occ + 1

        ablated = ablate_word(prompt, word, occ)
        if ablated == prompt:
            continue

        abl_emb = encode_text(model, device, ablated)
        abl_sim = cosine_sim(img_emb, abl_emb)
        saliency = baseline - abl_sim

        results.append({
            "word": word,
            "position": pos,
            "occurrence": occ,
            "saliency": round(saliency, 4),
            "ablated_sim": round(abl_sim, 4),
        })

    results.sort(key=lambda x: x["saliency"], reverse=True)
    # Attach rank (1 = highest saliency)
    for rank, item in enumerate(results, 1):
        item["rank"] = rank

    return results, round(baseline, 4)


# ── Δ localization ────────────────────────────────────────────────────────────

def get_content_words(prompt: str) -> list[str]:
    return [w.lower() for w in re.findall(r"\b\w+\b", prompt) if w.lower() not in STOPWORDS]


def locate_delta(source_prompt: str, followup_prompt: str, mr_id: str | None) -> dict:
    """
    Identify which words changed between source and followup.
    Returns:
      delta_words_source: words in source not in followup (removed/changed)
      delta_words_followup: words in followup not in source (added/changed)
      shared_words: words present in both (non-Δ)
    """
    src_words = set(get_content_words(source_prompt))
    fup_words = set(get_content_words(followup_prompt))

    delta_src = src_words - fup_words      # removed
    delta_fup = fup_words - src_words      # added
    shared = src_words & fup_words         # non-Δ

    return {
        "mr_id": str(mr_id),
        "delta_removed": sorted(delta_src),
        "delta_added": sorted(delta_fup),
        "shared_words": sorted(shared),
    }


# ── rank shift analysis (Step 2) ─────────────────────────────────────────────

def compute_rank_shifts(
    source_saliency: list[dict],
    followup_saliency: list[dict],
    shared_words: set[str],
) -> list[dict]:
    """
    For each non-Δ word present in both saliency lists,
    compute rank change: positive = rank got worse (higher number = less dominant),
    negative = rank improved (word became more dominant in followup).

    Returns list sorted by rank_delta ascending (biggest dominance gains first).
    """
    src_rank = {item["word"].lower(): item["rank"] for item in source_saliency}
    fup_rank = {item["word"].lower(): item["rank"] for item in followup_saliency}
    src_sal  = {item["word"].lower(): item["saliency"] for item in source_saliency}
    fup_sal  = {item["word"].lower(): item["saliency"] for item in followup_saliency}

    shifts = []
    for word in shared_words:
        if word not in src_rank or word not in fup_rank:
            continue
        r_src = src_rank[word]
        r_fup = fup_rank[word]
        rank_delta = r_fup - r_src  # negative = word rose in rank (more dominant in followup)
        shifts.append({
            "word": word,
            "source_rank": r_src,
            "followup_rank": r_fup,
            "rank_delta": rank_delta,         # negative = became more dominant
            "source_saliency": src_sal[word],
            "followup_saliency": fup_sal[word],
            "saliency_delta": round(fup_sal[word] - src_sal[word], 4),
        })

    # Sort by rank_delta ascending = biggest dominance gains (most negative) first
    shifts.sort(key=lambda x: x["rank_delta"])
    return shifts


# ── manifest loading ──────────────────────────────────────────────────────────

def load_manifests(manifest_paths: list[Path]) -> dict[tuple, dict]:
    """
    Returns dict keyed by (prompt_index, mr_id_str) → manifest record.
    mr_id is stored as string (None → "None").
    """
    records: dict[tuple, dict] = {}
    for path in manifest_paths:
        if not path.exists():
            print(f"[warn] manifest not found: {path}")
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("status") not in ("generated",):
                    continue
                key = (r["prompt_index"], str(r.get("mr_id")))
                records[key] = r
    return records


def resolve_image_path(record: dict, image_root: Path) -> Path | None:
    raw = record.get("image_path", "")
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # Try relative to image_root
    candidate = image_root / p
    if candidate.exists():
        return candidate
    # Try just filename under image_root recursively
    name = p.name
    for found in image_root.rglob(name):
        return found
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_CONFIGS.keys()), default="sdxl")
    parser.add_argument("--all-pairs", action="store_true",
                        help="Run on all pairs, not just inconsistent ones")
    parser.add_argument("--output", default=None,
                        help="Output JSONL path (default: data/results/stage2_{model}_saliency.jsonl)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process at most N pairs (for testing)")
    args = parser.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    out_path = Path(args.output) if args.output else ROOT / f"data/results/stage2_{args.model}_saliency.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Model    : {args.model}")
    print(f"Output   : {out_path}")

    # Load output_relation results
    relation_records = []
    with open(cfg["relation_file"]) as f:
        for line in f:
            r = json.loads(line)
            if args.all_pairs or not r.get("final_consistent", True):
                relation_records.append(r)
    print(f"Pairs to analyze: {len(relation_records)}")

    # Load manifests
    all_manifest_paths = (
        [cfg["source_manifest"]] + cfg["followup_manifest"]
        if isinstance(cfg["followup_manifest"], list)
        else [cfg["source_manifest"], cfg["followup_manifest"]]
    )
    manifest = load_manifests(list(set(all_manifest_paths)))
    print(f"Manifest entries (generated): {len(manifest)}")

    # Load CLIP
    print("Loading CLIP ViT-L/14...")
    model, preprocess, device = load_clip()
    print(f"  Device: {device}\n")

    processed = 0
    skipped = 0

    with open(out_path, "w") as out_f:
        for rel in relation_records:
            if args.limit and processed >= args.limit:
                break

            idx = rel["prompt_index"]
            mr_id = rel["mr_id"]

            src_rec = manifest.get((idx, "None"))
            fup_rec = manifest.get((idx, str(mr_id)))

            if not src_rec or not fup_rec:
                skipped += 1
                continue

            src_img = resolve_image_path(src_rec, cfg["image_root"])
            fup_img = resolve_image_path(fup_rec, cfg["image_root"])

            if not src_img or not fup_img:
                skipped += 1
                continue

            src_prompt = src_rec["prompt"]
            fup_prompt = fup_rec["prompt"]

            print(f"[{processed+1}] idx={idx} mr={mr_id} [{rel.get('stage','')}]")
            print(f"  SRC: {src_prompt[:80]}")
            print(f"  FUP: {fup_prompt[:80]}")

            # Step 1: compute saliency for both
            src_sal, src_baseline = compute_saliency(src_img, src_prompt, model, preprocess, device)
            fup_sal, fup_baseline = compute_saliency(fup_img, fup_prompt, model, preprocess, device)

            # Step 2: locate Δ
            delta = locate_delta(src_prompt, fup_prompt, mr_id)
            shared = set(delta["shared_words"])

            # Step 3: compute rank shifts for non-Δ words
            rank_shifts = compute_rank_shifts(src_sal, fup_sal, shared)

            # Identify top dominance-gain suspects (rank decreased = word rose)
            # rank_delta < 0 means followup_rank < source_rank (word rose in dominance)
            suspects = [s for s in rank_shifts if s["rank_delta"] < 0]

            record = {
                # identity
                "prompt_index": idx,
                "mr_id": str(mr_id),
                "category": rel.get("category"),
                "prompt_type": rel.get("prompt_type"),
                "anchor_cue": rel.get("anchor_cue"),
                "stage": rel.get("stage"),
                "final_consistent": rel.get("final_consistent"),
                # prompts
                "source_prompt": src_prompt,
                "followup_prompt": fup_prompt,
                # images
                "source_image": str(src_img),
                "followup_image": str(fup_img),
                # layer results from Stage 1
                "layer3_clip_similarity": rel.get("layer3_clip_similarity"),
                # saliency
                "source_baseline_sim": src_baseline,
                "followup_baseline_sim": fup_baseline,
                "source_saliency": src_sal,
                "followup_saliency": fup_sal,
                # Δ localization
                "delta": delta,
                # rank shift analysis (non-Δ words sorted by dominance gain)
                "rank_shifts": rank_shifts,
                "dominance_gain_suspects": suspects,
            }

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed += 1

            # Print top shifts
            print(f"  Source baseline: {src_baseline:.4f}  |  Followup baseline: {fup_baseline:.4f}")
            print(f"  Δ removed: {delta['delta_removed']}  |  added: {delta['delta_added']}")
            if suspects:
                print(f"  Top dominance-gain suspects (non-Δ words that rose in rank):")
                for s in suspects[:3]:
                    print(f"    '{s['word']}': rank {s['source_rank']}→{s['followup_rank']} "
                          f"(Δrank={s['rank_delta']})  sal {s['source_saliency']:+.4f}→{s['followup_saliency']:+.4f}")
            print()

    print(f"Done: {processed} processed, {skipped} skipped")
    print(f"Results → {out_path}")


if __name__ == "__main__":
    main()
