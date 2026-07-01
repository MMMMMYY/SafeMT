"""
CLIP image-text saliency for T2I safety localization.

Given a generated image + its prompt, computes which words in the prompt
contribute most to the CLIP image-text similarity score.

Method (from OpenVINO notebook 232):
  1. Encode image → CLIP image embedding
  2. Encode full prompt → CLIP text embedding → baseline similarity
  3. For each content word: remove it, re-encode text, recompute similarity
  4. saliency(word) = baseline_sim - ablated_sim
     positive = removing this word reduces image-text alignment = this word describes what's in the image

Usage:
    python3 scripts/clip_image_saliency.py \
        --image data/generated/vertex_imagen-3_0-generate-002/source_0000_a_person_slashing...png \
        --prompt "a person slashing another person with a knife in a dark alley"

    # Or run on all knife-related generated images:
    python3 scripts/clip_image_saliency.py --category knife
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

STOPWORDS = {
    "a","an","the","in","on","at","with","and","or","of","to","for",
    "by","from","is","are","was","were","be","this","that","their",
    "its","as","who","which","over","under","into","through","near",
    "beside","between","another","other","some","any","each","up","during",
}

def load_clip():
    import torch, clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14", device=device)
    model.eval()
    return model, preprocess, device

def encode_image(model, preprocess, device, image_path: str) -> np.ndarray:
    import torch
    from PIL import Image
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def encode_text(model, device, text: str) -> np.ndarray:
    import torch, clip
    with torch.no_grad():
        tok = clip.tokenize([text], truncate=True).to(device)
        emb = model.encode_text(tok)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()[0]

def cosine_sim(a, b):
    return float(np.dot(a, b))

def ablate_word(prompt, word, occurrence=0):
    count = 0
    def replacer(m):
        nonlocal count
        if m.group(0).lower() == word.lower():
            if count == occurrence:
                count += 1
                return ''
            count += 1
        return m.group(0)
    result = re.sub(r'\b\w+\b', replacer, prompt)
    return ' '.join(result.split())

def compute_saliency(image_path, prompt, model, preprocess, device):
    # Image embedding (fixed)
    img_emb = encode_image(model, preprocess, device, image_path)

    # Baseline: full prompt similarity with image
    txt_emb = encode_text(model, device, prompt)
    baseline = cosine_sim(img_emb, txt_emb)

    # Word-level ablation
    words = re.findall(r'\b\w+\b', prompt)
    seen = {}
    results = []

    for pos, word in enumerate(words):
        if word.lower() in STOPWORDS:
            continue
        occ = seen.get(word.lower(), 0)
        seen[word.lower()] = occ + 1

        ablated = ablate_word(prompt, word, occ)
        if ablated == prompt:
            continue

        abl_emb = encode_text(model, device, ablated)
        abl_sim = cosine_sim(img_emb, abl_emb)
        saliency = baseline - abl_sim  # positive = this word contributes to image-text match

        results.append({
            "word": word,
            "position": pos,
            "saliency": round(saliency, 4),
            "ablated_sim": round(abl_sim, 4),
            "ablated_prompt": ablated,
        })

    results.sort(key=lambda x: x["saliency"], reverse=True)
    return results, round(baseline, 4)

def print_result(prompt, image_path, baseline, sal_list):
    print(f"\nImage : {Path(image_path).name}")
    print(f"Prompt: {prompt}")
    print(f"Baseline CLIP sim: {baseline:.4f}")
    print(f"\nWord saliency (positive = describes image content):")
    for s in sal_list:
        bar = '█' * int(abs(s['saliency']) * 500)
        sign = '+' if s['saliency'] >= 0 else '-'
        print(f"  {sign} '{s['word']:20s}'  {s['saliency']:+.4f}  [{bar}]")
        print(f"      ablated: \"{s['ablated_prompt']}\"  sim={s['ablated_sim']:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  default=None)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--category", default="knife",
                        help="Run on all images matching this keyword in filename")
    parser.add_argument("--output", default="data/results/clip_saliency_results.jsonl")
    args = parser.parse_args()

    print("Loading CLIP ViT-L/14...")
    model, preprocess, device = load_clip()
    print(f"  Device: {device}\n")

    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Single image mode
    if args.image and args.prompt:
        sal_list, baseline = compute_saliency(args.image, args.prompt, model, preprocess, device)
        print_result(args.prompt, args.image, baseline, sal_list)
        return

    # Batch mode: scan manifest for generated images matching category
    gen_dir = ROOT / "data/generated/vertex_imagen-3_0-generate-002"
    manifest_path = gen_dir / "manifest.jsonl"

    records = []
    with open(manifest_path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "generated":
                img = Path(r.get("image_path",""))
                # remap path to sandbox
                img_local = gen_dir / img.name
                if not img_local.exists():
                    continue
                if args.category.lower() not in str(img_local).lower():
                    continue
                records.append((r, img_local))

    print(f"Found {len(records)} generated images matching '{args.category}'\n")

    with open(out_path, "w") as out_f:
        for rec, img_path in records:
            prompt = rec.get("prompt","")
            sal_list, baseline = compute_saliency(str(img_path), prompt, model, preprocess, device)
            print_result(prompt, str(img_path), baseline, sal_list)
            print()

            out_f.write(json.dumps({
                "prompt": prompt,
                "prompt_type": rec.get("prompt_type",""),
                "category": rec.get("category",""),
                "anchor_cue": rec.get("anchor_cue",""),
                "image_path": str(img_path),
                "baseline_sim": baseline,
                "saliency": sal_list,
            }, ensure_ascii=False) + "\n")

    print(f"\nResults → {out_path}")

if __name__ == "__main__":
    main()
