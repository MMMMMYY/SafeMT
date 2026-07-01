"""
Stage 1: Generate images for llmorph MR follow-up prompts.
Mirrors run_vertex_source_batch.py but reads followup_inputs files.

Usage:
    # Vertex (Imagen):
    python3 scripts/run_followup_batch.py \
        --provider vertex \
        --mr-id 51 \
        --project-id YOUR_PROJECT_ID

    # Gemini:
    python3 scripts/run_followup_batch.py \
        --provider gemini \
        --mr-id 51

    # All MRs at once:
    for MR in 51 8 19 155; do
        python3 scripts/run_followup_batch.py --provider gemini --mr-id $MR
    done

Output images: data/generated/{provider}/followup_mr{mr_id}/
Manifest:      data/generated/{provider}/followup_mr{mr_id}/manifest.jsonl

Manifest fields match run_vertex_source_batch.py format, plus:
  mr_id, prompt_index, prompt_type, category, anchor_cue
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
CHATGPT_PY  = ROOT / "ChatGPT.py"
LLMORPH_DIR = ROOT / "llmorph" / "data" / "prompt-mutation"
METADATA    = ROOT / "data" / "interim" / "source_prompt_metadata.jsonl"


def load_followup_prompts(mr_id: str) -> list[str]:
    path = LLMORPH_DIR / "followup_inputs" / f"followup_inputs__prompt_mutation__{mr_id}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Followup file not found: {path}\n"
            f"Run llmorph first: cd llmorph && python3 src/mt_main.py"
        )
    data = json.loads(path.read_text())
    prompts = []
    for item in data:
        # Format: [[prompt_string]] or [prompt_string]
        if isinstance(item, list) and item:
            inner = item[0]
            if isinstance(inner, list) and inner:
                prompts.append(str(inner[0]).strip().strip('"'))
            elif isinstance(inner, str):
                prompts.append(inner.strip().strip('"'))
            else:
                prompts.append("")
        else:
            prompts.append("")
    return prompts


def load_source_prompts() -> list[str]:
    path = LLMORPH_DIR / "source_inputs" / "source_inputs_v2.json"
    data = json.loads(path.read_text())
    return [item[0] if isinstance(item, list) and item else "" for item in data]


def load_metadata() -> dict[int, dict]:
    meta = {}
    if not METADATA.exists():
        return meta
    with open(METADATA) as f:
        for line in f:
            r = json.loads(line)
            meta[r["index"]] = r
    return meta


def sanitize(text: str, max_len: int = 70) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:max_len].strip("_")


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def is_blocked(returncode: int, stderr: str, output_path: Path) -> tuple[bool, str]:
    if returncode != 0:
        s = stderr.lower()
        if any(k in s for k in ["no predictions", "response: {}", "block", "safety", "policy"]):
            return True, "safety_blocked"
        return True, f"error: {stderr[:120]}"
    if not output_path.exists() or output_path.stat().st_size < 200:
        return True, "empty_output"
    return False, ""


def generate_one(prompt: str, output_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Run ChatGPT.py for one prompt. Returns (status, detail)."""
    if output_path.exists() and not args.overwrite:
        return "skipped", "exists"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_exe, str(CHATGPT_PY),
        "--provider", args.provider,
        "--model",    args.model,
        "--prompt",   prompt,
        "--reference", "",
        "--output",   str(output_path),
        "--aspect-ratio", "1:1",
    ]
    if args.provider == "vertex":
        if args.project_id:
            cmd += ["--project-id", args.project_id]
        if args.location:
            cmd += ["--location", args.location]
        if args.credentials_file:
            cmd += ["--credentials-file", args.credentials_file]

    for attempt in range(3):
        if attempt > 0:
            time.sleep(45)
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            break
        if "429" in result.stderr and attempt < 2:
            continue
        blocked, detail = is_blocked(result.returncode, result.stderr, output_path)
        return ("blocked" if blocked else "error"), detail
    blocked, detail = is_blocked(result.returncode, result.stderr, output_path)
    if blocked:
        return "blocked", detail
    return "generated", ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider",   choices=["vertex", "gemini", "openai", "sdxl"],
                        default="gemini")
    parser.add_argument("--mr-id",      required=True,
                        help="MR id to process, e.g. 51")
    parser.add_argument("--model",      default=None)
    parser.add_argument("--limit",      type=int, default=None)
    parser.add_argument("--start-index",type=int, default=0)
    parser.add_argument("--overwrite",  action="store_true")
    parser.add_argument("--python-exe", default="python3")
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--location",   default="us-central1")
    parser.add_argument("--credentials-file", default=None)
    parser.add_argument("--output-root",
                        default=str(ROOT / "data" / "generated"))
    args = parser.parse_args()

    # Default models
    defaults = {
        "vertex": "imagen-3.0-generate-002",
        "gemini": "gemini-2.5-flash-image",
        "openai": "gpt-image-1",
    }
    if not args.model:
        args.model = defaults.get(args.provider, "")

    # Paths
    out_dir       = Path(args.output_root) / args.provider / f"followup_mr{args.mr_id}"
    manifest_path = out_dir / "manifest.jsonl"

    # Load data
    followup_prompts = load_followup_prompts(args.mr_id)
    source_prompts   = load_source_prompts()
    metadata         = load_metadata()

    end = len(followup_prompts) if args.limit is None \
          else min(len(followup_prompts), args.start_index + args.limit)
    prompts_to_run = list(enumerate(followup_prompts))[args.start_index:end]

    print(f"Provider : {args.provider} / {args.model}")
    print(f"MR       : {args.mr_id}")
    print(f"Output   : {out_dir}")
    print(f"Prompts  : {len(prompts_to_run)}")
    print()

    stats = {"generated": 0, "blocked": 0, "skipped": 0, "error": 0}

    for job_num, (idx, followup_prompt) in enumerate(prompts_to_run, 1):
        if not followup_prompt:
            continue

        source_prompt = source_prompts[idx] if idx < len(source_prompts) else ""
        meta          = metadata.get(idx, {})
        ptype         = meta.get("prompt_type", "unknown")
        category      = meta.get("category", "")
        anchor_cue    = meta.get("anchor_cue", "")

        fname      = f"mr{args.mr_id}_{idx:04d}_{sanitize(followup_prompt)}.png"
        img_path   = out_dir / fname

        print(f"[{job_num}/{len(prompts_to_run)}] #{idx:03d} [{ptype}|{category}]")
        print(f"  src: {source_prompt[:70]}")
        print(f"  mr{args.mr_id}: {followup_prompt[:70]}")

        started = datetime.now(timezone.utc).isoformat()
        t0      = time.time()
        status, detail = generate_one(followup_prompt, img_path, args)
        elapsed = round(time.time() - t0, 2)

        stats[status] = stats.get(status, 0) + 1
        icon = {"generated":"✓","blocked":"✗","skipped":"-"}.get(status,"!")
        print(f"  {icon} {status}  ({elapsed}s)" + (f"  [{detail}]" if detail else ""))

        append_manifest(manifest_path, {
            "prompt_index":   idx,
            "mr_id":          args.mr_id,
            "prompt_type":    ptype,
            "category":       category,
            "anchor_cue":     anchor_cue,
            "source_prompt":  source_prompt,
            "followup_prompt": followup_prompt,
            "provider":       args.provider,
            "model":          args.model,
            "image_path":     str(img_path),
            "status":         status,
            "detail":         detail,
            "started_at":     started,
            "elapsed_s":      elapsed,
        })
        if status not in ("skipped",):
            time.sleep(2.0)

    print(f"\nDone: {stats}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
