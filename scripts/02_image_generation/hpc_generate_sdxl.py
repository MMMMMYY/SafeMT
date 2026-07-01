"""
Standalone SDXL batch generation for HPC cluster — no SwarmUI dependency.
Uses HuggingFace diffusers directly.

Usage:
    python3 hpc_generate_sdxl.py \
        --jobs hpc_sdxl_jobs.json \
        --output-dir ./output \
        --start 0 --end 1350
"""

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path


def sanitize(text: str, max_len: int = 60) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:max_len].strip("_")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--guidance-scale", type=float, default=7.0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch
    from diffusers import StableDiffusionXLPipeline

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"manifest_{args.start}_{args.end}.jsonl"

    print(f"Loading {args.model} ...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        args.model, torch_dtype=torch.float16, use_safetensors=True
    )
    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    print("Model loaded.", flush=True)

    jobs = json.loads(Path(args.jobs).read_text())
    end = args.end if args.end is not None else len(jobs)
    jobs = jobs[args.start:end]
    print(f"Processing jobs [{args.start}:{end}] = {len(jobs)} jobs", flush=True)

    stats = {"generated": 0, "blocked": 0, "error": 0, "skipped": 0}
    for i, job in enumerate(jobs, 1):
        idx = job["prompt_index"]
        mr_id = job["mr_id"]
        prompt = job["prompt"]
        fname = (
            f"source_{idx:04d}_{sanitize(prompt)}.png"
            if mr_id is None
            else f"mr{mr_id}_{idx:04d}_{sanitize(prompt)}.png"
        )
        img_path = out_dir / fname

        if img_path.exists() and not args.overwrite:
            stats["skipped"] += 1
            continue

        label = f"MR-{mr_id}" if mr_id else "source"
        print(f"[{i}/{len(jobs)}] {label} #{idx:03d} [{job['prompt_type']}] {prompt[:60]}", flush=True)

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status, detail = "generated", ""
        try:
            result = pipe(
                prompt=prompt,
                num_inference_steps=args.steps,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance_scale,
            )
            image = result.images[0]
            if hasattr(result, "nsfw_content_detected") and result.nsfw_content_detected and result.nsfw_content_detected[0]:
                status, detail = "blocked", "NSFW safety checker triggered"
            else:
                image.save(img_path)
        except Exception as e:
            status, detail = "error", str(e)[:400]

        elapsed = round(time.time() - t0, 2)
        stats[status] += 1
        print(f"  -> {status} ({elapsed}s) {detail[:60]}", flush=True)

        with open(manifest_path, "a") as f:
            f.write(json.dumps({
                "prompt_index": idx, "mr_id": mr_id, "prompt_type": job["prompt_type"],
                "category": job["category"], "anchor_cue": job["anchor_cue"],
                "prompt": prompt, "provider": "sdxl-hpc", "model": args.model,
                "image_path": str(img_path), "status": status, "detail": detail,
                "started_at": started, "elapsed_s": elapsed,
            }, ensure_ascii=False) + "\n")

    print(f"\nDone: {stats}")


if __name__ == "__main__":
    main()
