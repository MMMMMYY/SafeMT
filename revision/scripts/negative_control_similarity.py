"""
Negative control for the Layer-3 semantic threshold.
Samples random pairs of generated images whose SOURCE PROMPTS use different anchors
AND different safety categories (i.e., unrelated images), computes CLIP ViT-L/14
image-image cosine similarity with the exact same pipeline as output_relation.py,
and reports the distribution.

Usage:
    python3 revision/scripts/negative_control_similarity.py --n-pairs 1000
"""
import argparse, json, random, statistics
from pathlib import Path
import numpy as np
import torch, clip
from PIL import Image

# Section V-B threshold interpretation: unrelated-image CLIP negative control.
from package_paths import ROOT, GENERATED, MAIN_MANIFESTS, original_image
MANIFESTS = MAIN_MANIFESTS
def resolve(raw, mpath):
    return original_image(raw)

def load_images(paths):
    out, seen = [], set()
    for m in paths:
        for line in open(m):
            r = json.loads(line)
            if r.get("status") != "generated":
                continue
            p = resolve(r["image_path"], m)
            if p is None or str(p) in seen:
                continue
            seen.add(str(p))
            out.append({"path": str(p), "anchor": r.get("anchor_cue"), "category": r.get("category")})
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pairs", type=int, default=1000, help="pairs per system")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    model, preprocess = clip.load("ViT-L/14", device=args.device, jit=False)
    model.eval()
    cache = {}

    @torch.no_grad()
    def enc(p):
        if p not in cache:
            x = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(args.device)
            f = model.encode_image(x).float()
            cache[p] = f / f.norm(dim=-1, keepdim=True)
        return cache[p]

    summary, all_sims = {}, []
    for sysname, paths in MANIFESTS.items():
        imgs = load_images(paths)
        sims, tries = [], 0
        while len(sims) < args.n_pairs and tries < args.n_pairs * 50:
            tries += 1
            a, b = rng.sample(imgs, 2)
            if a["anchor"] == b["anchor"] or a["category"] == b["category"]:
                continue  # keep only clearly unrelated pairs
            sims.append((enc(a["path"]) @ enc(b["path"]).T).item())
        all_sims += sims
        s = np.array(sims)
        summary[sysname] = {
            "n_pairs": len(s), "mean": round(float(s.mean()), 4), "std": round(float(s.std()), 4),
            "p95": round(float(np.percentile(s, 95)), 4), "p99": round(float(np.percentile(s, 99)), 4),
            "max": round(float(s.max()), 4), "frac_ge_0.70": round(float((s >= 0.70).mean()), 4),
        }
        print(sysname, summary[sysname], flush=True)
    s = np.array(all_sims)
    summary["all"] = {"n_pairs": len(s), "mean": round(float(s.mean()), 4),
                      "p95": round(float(np.percentile(s, 95)), 4), "p99": round(float(np.percentile(s, 99)), 4),
                      "max": round(float(s.max()), 4), "frac_ge_0.70": round(float((s >= 0.70).mean()), 4)}
    print("all", summary["all"])
    out = GENERATED / "negative_control_similarity.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(out, "w"), indent=2)
    print("saved to", out)

if __name__ == "__main__":
    main()
