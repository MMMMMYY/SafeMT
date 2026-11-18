"""
Repeated-generation collection for the controls in Tables V--VII.

Re-generates every SOURCE prompt k times under the configured generation settings,
with fresh API calls (no seed is exposed by GPT-Image-1 / Gemini). Output is
kept separate from the original run so nothing in data/generated/safemt is touched.

Writes:
  data/generated/safemt_reps/{model_tag}/source_{idx:04d}_rep{r}_{slug}.png
  data/generated/safemt_reps/{model_tag}/manifest_reps.jsonl   (one line per attempt; has "rep")

Re-uses the exact provider call + retry + blocked-detection logic from
run_generation_batch.py so generation conditions match the original experiment.

Usage (inference dependencies installed, repository root):
  python3 revision/scripts/run_repeated_generation.py --provider openai --reps 3 --quality medium
  python3 revision/scripts/run_repeated_generation.py --provider gemini-vertex --reps 3 \
        --project-id YOUR_PROJECT --credentials-file creds.json --min-interval 13
  # smoke test
  python3 revision/scripts/run_repeated_generation.py --provider openai --reps 3 --limit 3
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generation_backend import (  # noqa: E402
    load_source_prompts, load_metadata, sanitize, append_manifest, generate_one, ROOT,
)

from package_paths import SOURCE_PROMPT_COUNT, REPETITION_COUNT

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--provider", choices=["openai", "gemini-vertex", "sdxl"], required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--reps", type=int, default=REPETITION_COUNT, choices=[REPETITION_COUNT])
    p.add_argument("--rep-start", type=int, default=1, choices=[1], help="first rep index (1-based)")
    p.add_argument("--limit", type=int, default=None, help="only first N source prompts (smoke test)")
    p.add_argument("--prompt-indices", nargs="*", type=int, default=None)
    p.add_argument("--only-errors", action="store_true",
                   help="re-run only (prompt, rep) slots whose latest manifest status is error")
    p.add_argument("--output-root", default=str(ROOT / "revision" / "generated" / "repetitions"))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Show the planned request count without querying a model.")
    # pass-through args used by run_generation_batch helpers
    p.add_argument("--python-exe", default="python3")
    p.add_argument("--swarm-url", default="http://127.0.0.1:7801")
    p.add_argument("--aspect-ratio", default="1:1")
    p.add_argument("--quality", default="medium")
    p.add_argument("--max-attempts", type=int, default=3, dest="max_attempts")
    p.add_argument("--min-interval", type=float, default=0, dest="min_interval")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--project-id", default=None,
                   help="GCP project id for gemini-vertex; defaults to contents of gemini_project.txt")
    p.add_argument("--location", default="us-central1")
    p.add_argument("--credentials-file", default=None)
    args = p.parse_args()
    if args.provider == "gemini-vertex" and not args.project_id:
        pf = ROOT / "gemini_project.txt"
        if pf.exists():
            args.project_id = pf.read_text().strip()

    default_models = {
        "sdxl": "OfficialStableDiffusion/sd_xl_base_1.0.safetensors",
        "gemini-vertex": "gemini-2.5-flash-image",
        "openai": "gpt-image-1",
    }
    model = args.model or default_models[args.provider]
    model_tag = f"{args.provider}_{model.split('/')[-1].replace('.', '_')[:30]}"
    out_dir = Path(args.output_root) / model_tag
    manifest_path = out_dir / "manifest_reps.jsonl"

    prompts = load_source_prompts()
    assert len(prompts) == SOURCE_PROMPT_COUNT
    meta = load_metadata()
    indices = list(range(len(prompts)))
    if args.prompt_indices:
        indices = args.prompt_indices
    if args.limit:
        indices = indices[: args.limit]

    error_slots = None
    if args.only_errors and manifest_path.exists():
        final = {}
        for line in open(manifest_path):
            rec = json.loads(line)
            k = (rec["prompt_index"], rec["rep"])
            if k not in final or rec["status"] in ("generated", "blocked") or final[k] == "error":
                final[k] = rec["status"]
        error_slots = {k for k, st in final.items() if st == "error"}
        print(f"only-errors: {len(error_slots)} slots to retry")

    jobs = []
    for i in indices:
        if not prompts[i]:
            continue
        for r in range(args.rep_start, args.rep_start + args.reps):
            if error_slots is not None and (i, r) not in error_slots:
                continue
            fname = f"source_{i:04d}_rep{r}_{sanitize(prompts[i])}.png"
            jobs.append((i, r, prompts[i], out_dir / fname))

    print(f"Provider : {args.provider} / {model}\nOutput   : {out_dir}\nJobs     : {len(jobs)} "
          f"({len(indices)} prompts x {args.reps} reps)\n")

    if args.dry_run:
        return

    stats = {}
    for n, (idx, rep, prompt, img_path) in enumerate(jobs, 1):
        m = meta.get(idx, {})
        print(f"[{n}/{len(jobs)}] #{idx:03d} rep{rep} [{m.get('prompt_type','?')}] {prompt[:70]}")
        t0 = time.time()
        status, detail = generate_one(args.provider, model, prompt, img_path, args)
        elapsed = round(time.time() - t0, 2)
        stats[status] = stats.get(status, 0) + 1
        print(f"  {status} ({elapsed}s){'  ['+detail[-120:]+']' if detail else ''}")
        append_manifest(manifest_path, {
            "prompt_index": idx, "rep": rep, "mr_id": None,
            "prompt_type": m.get("prompt_type", ""), "category": m.get("category", ""),
            "anchor_cue": m.get("anchor_cue", ""), "prompt": prompt,
            "provider": args.provider, "model": model, "quality": args.quality if args.provider == "openai" else None,
            "image_path": str(img_path), "status": status, "detail": detail,
            "elapsed_s": elapsed,
        })
        if status != "skipped":
            time.sleep(2.0)
    print(f"\nDone: {stats}\nManifest -> {manifest_path}")


if __name__ == "__main__":
    main()
