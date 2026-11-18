"""
SDXL repeated generation with diffusers (same settings as hpc_generate_sdxl.py:
SDXL base 1.0, 1024x1024, 20 steps, cfg 7.0) but with an EXPLICIT per-job seed.

Usage:
  python3 generate_sdxl_repetitions.py --jobs hpc_sdxl_rep_jobs.json --output-dir ./output_reps --start 0 --end 810
"""
import argparse, json, re, time
from pathlib import Path
from package_paths import SOURCE_PROMPT_COUNT, REPETITION_COUNT


def sanitize(t, n=60):
    return re.sub(r"[^a-z0-9]+", "_", t.lower())[:n].strip("_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=None)
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--guidance-scale", type=float, default=7.0)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    out = Path(a.output_dir); out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest_reps.jsonl"
    pipe = StableDiffusionXLPipeline.from_pretrained(a.model, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    jobs = json.loads(Path(a.jobs).read_text())
    assert len(jobs) == SOURCE_PROMPT_COUNT * REPETITION_COUNT
    assert {(j["prompt_index"],j["rep"]) for j in jobs} == {(i,r) for i in range(SOURCE_PROMPT_COUNT) for r in range(1,REPETITION_COUNT+1)}
    jobs = jobs[a.start: a.end if a.end is not None else len(jobs)]
    stats = {}
    for n, j in enumerate(jobs, 1):
        idx, rep, seed, prompt = j["prompt_index"], j["rep"], j["seed"], j["prompt"]
        img = out / f"source_{idx:04d}_rep{rep}_{sanitize(prompt)}.png"
        if img.exists() and not a.overwrite:
            stats["skipped"] = stats.get("skipped", 0) + 1; continue
        print(f"[{n}/{len(jobs)}] #{idx:03d} rep{rep} seed={seed} {prompt[:60]}", flush=True)
        t0 = time.time()
        status, detail = "generated", ""
        try:
            g = torch.Generator(device="cuda").manual_seed(seed)
            res = pipe(prompt=prompt, num_inference_steps=a.steps, width=a.width, height=a.height,
                       guidance_scale=a.guidance_scale, generator=g)
            res.images[0].save(img)
        except Exception as e:
            status, detail = "error", str(e)[:400]
        stats[status] = stats.get(status, 0) + 1
        with open(manifest, "a") as f:
            f.write(json.dumps({"prompt_index": idx, "rep": rep, "seed": seed, "mr_id": None,
                                "prompt_type": j["prompt_type"], "category": j["category"], "anchor_cue": j["anchor_cue"],
                                "prompt": prompt, "provider": "sdxl", "model": a.model,
                                "steps": a.steps, "guidance_scale": a.guidance_scale,
                                "image_path": str(img), "status": status, "detail": detail,
                                "elapsed_s": round(time.time() - t0, 2)}, ensure_ascii=False) + "\n")
    print("Done:", stats)


if __name__ == "__main__":
    main()
