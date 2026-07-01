"""
Retry only the manifest entries with status=='error' for a given model output dir.
Reuses generation helpers from run_generation_batch.py.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_generation_batch import (
    ROOT, load_source_prompts, load_followup_prompts, load_metadata,
    sanitize, append_manifest, generate_one, FOLLOWUP_DIR,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", default="vertex")
    parser.add_argument("--model", default="imagen-3.0-generate-002")
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default="python3")
    parser.add_argument("--swarm-url", default="http://127.0.0.1:7801")
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--quality", default="medium")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--credentials-file", default=None)
    parser.add_argument("--overwrite", action="store_true", default=True)
    args = parser.parse_args()

    model_tag = f"{args.provider}_{args.model.split('/')[-1].replace('.', '_')[:30]}"
    out_dir = Path(args.output_root) / model_tag
    manifest_path = out_dir / "manifest.jsonl"

    records = [json.loads(l) for l in open(manifest_path)]
    errors = [r for r in records if r["status"] == "error"]
    print(f"Found {len(errors)} error records to retry")

    source_prompts = load_source_prompts()
    metadata = load_metadata()
    followup_cache = {}

    import time
    from datetime import datetime, timezone
    stats = {"generated": 0, "blocked": 0, "error": 0}
    for i, err in enumerate(errors, 1):
        idx = err["prompt_index"]
        mr_id = err["mr_id"]
        if mr_id is None:
            prompt = source_prompts[idx]
            fname = f"source_{idx:04d}_{sanitize(prompt)}.png"
        else:
            if mr_id not in followup_cache:
                followup_cache[mr_id] = load_followup_prompts(mr_id)
            prompt = followup_cache[mr_id][idx]
            fname = f"mr{mr_id}_{idx:04d}_{sanitize(prompt)}.png"
        img_path = out_dir / fname

        label = f"MR-{mr_id}" if mr_id else "source"
        meta = metadata.get(idx, {})
        ptype = meta.get("prompt_type", "unknown")
        print(f"[{i}/{len(errors)}] {label} #{idx:03d} [{ptype}] {prompt[:60]}")

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status, detail = generate_one(args.provider, args.model, prompt, img_path, args)
        elapsed = round(time.time() - t0, 2)
        stats[status] = stats.get(status, 0) + 1
        icon = {"generated": "+", "blocked": "x", "error": "!"}.get(status, "?")
        print(f"  {icon} {status} ({elapsed}s) {detail[:60] if detail else ''}")

        append_manifest(manifest_path, {
            "prompt_index": idx, "mr_id": mr_id, "prompt_type": ptype,
            "category": meta.get("category", ""), "anchor_cue": meta.get("anchor_cue", ""),
            "prompt": prompt, "provider": args.provider, "model": args.model,
            "image_path": str(img_path), "status": status, "detail": detail,
            "started_at": started, "elapsed_s": elapsed, "retry_of_error": True,
        })
        if status != "skipped":
            time.sleep(2.0)

    print(f"\nRetry done: {stats}")


if __name__ == "__main__":
    main()
