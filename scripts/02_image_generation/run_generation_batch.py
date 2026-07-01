"""
Unified T2I batch generation script.
Handles both source prompts and Stage 1 MR follow-up prompts.

Reads:
  - data/prompt-mutation/source_inputs/source_inputs_v2.json  (source prompts)
  - data/prompt-mutation/followup_inputs/followup_inputs__prompt_mutation__{MR}.json

Writes images to:
  - data/generated/{model_tag}/source_{idx:04d}.png
  - data/generated/{model_tag}/mr{mr_id}_{idx:04d}.png

Writes a manifest JSONL for every image:
  - data/generated/{model_tag}/manifest.jsonl

Usage examples:
    # SDXL via SwarmUI (local)
    python3 scripts/run_generation_batch.py --provider sdxl --limit 20

    # Vertex Imagen
    python3 scripts/run_generation_batch.py --provider vertex --project-id MY_PROJECT

    # Gemini
    python3 scripts/run_generation_batch.py --provider gemini

    # Only source prompts, first 30
    python3 scripts/run_generation_batch.py --provider sdxl --prompt-set source --limit 30

    # Only followup for MR-51
    python3 scripts/run_generation_batch.py --provider sdxl --prompt-set followup --mr-ids 51
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLMORPH_DIR = ROOT / "llmorph" / "data" / "prompt-mutation"
SOURCE_JSON = LLMORPH_DIR / "source_inputs" / "source_inputs_safemt.json"
FOLLOWUP_DIR = LLMORPH_DIR / "followup_inputs"
METADATA_JSONL = ROOT / "data" / "interim" / "dataset_construction" / "full_dataset_metadata.jsonl"
CHATGPT_PY = ROOT / "ChatGPT.py"

MR_IDS = ["51", "8", "204", "155", "artistic_medium", "photographic_style", "environment_details", "camera_perspective"]


# ── helpers ───────────────────────────────────────────────────────────────────

def load_source_prompts() -> list[str]:
    data = json.loads(SOURCE_JSON.read_text())
    return [item[0] if isinstance(item, list) and item else "" for item in data]


def load_followup_prompts(mr_id: str) -> list[str]:
    path = FOLLOWUP_DIR / f"followup_inputs__prompt_mutation__{mr_id}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text())
    out = []
    for item in data:
        if isinstance(item, list) and item and isinstance(item[0], list):
            out.append(item[0][0] if item[0] else "")
        elif isinstance(item, list) and item and isinstance(item[0], str):
            out.append(item[0])
        else:
            out.append("")
    return out


def load_metadata() -> dict[int, dict]:
    meta = {}
    if not METADATA_JSONL.exists():
        return meta
    with open(METADATA_JSONL) as f:
        for line in f:
            r = json.loads(line)
            meta[r["index"]] = r
    return meta


def sanitize(text: str, max_len: int = 60) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "_", text.lower())[:max_len].strip("_")


def append_manifest(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── provider calls ────────────────────────────────────────────────────────────

_last_call_time = [0.0]


def _throttle(min_interval: float) -> None:
    """Block until at least min_interval seconds have passed since the last call."""
    if min_interval <= 0:
        return
    now = time.time()
    wait = _last_call_time[0] + min_interval - now
    if wait > 0:
        time.sleep(wait)
    _last_call_time[0] = time.time()


def call_chatgpt_py(provider: str, model: str, prompt: str,
                    output_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Call ChatGPT.py wrapper. Returns (status, detail)."""
    cmd = [
        args.python_exe, str(CHATGPT_PY),
        "--provider", provider,
        "--model", model,
        "--prompt", prompt,
        "--reference", "",
        "--output", str(output_path),
    ]
    if args.aspect_ratio:
        cmd += ["--aspect-ratio", args.aspect_ratio]
    if provider == "openai":
        cmd += ["--quality", args.quality]
    if provider in ("vertex", "gemini-vertex"):
        if args.project_id:
            cmd += ["--project-id", args.project_id]
        if args.location:
            cmd += ["--location", args.location]
        if args.credentials_file:
            cmd += ["--credentials-file", args.credentials_file]
    max_attempts = getattr(args, "max_attempts", 3)
    min_interval = getattr(args, "min_interval", 0)
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(min(30 * (2 ** (attempt - 1)), 180))  # exponential backoff: 30s, 60s, 120s, capped at 180s
        _throttle(min_interval)
        result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode == 0:
            break
        if "429" in result.stderr and attempt < max_attempts - 1:
            continue
        err = result.stderr.lower()
        if any(k in err for k in [
            "block", "safety", "policy", "refus", "filtered", "no predictions",
            "cannot fulfill", "can't fulfill", "i cannot generate", "i can't generate",
            "unable to generate", "i'm unable to", "i am unable to",
            "ethical guidelines", "against my guidelines", "i cannot create", "i can't create",
            "i cannot assist", "i can't assist", "no image was returned",
        ]):
            return "blocked", result.stderr
        return "error", result.stderr
    if not output_path.exists() or output_path.stat().st_size < 100:
        return "blocked", "empty or missing output"
    return "generated", ""


def call_swarm(prompt: str, output_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Call local SwarmUI API."""
    from urllib.request import Request, urlopen
    from urllib.error import URLError
    import base64

    session_url = f"{args.swarm_url}/API/GetNewSession"
    try:
        req = Request(session_url, data=b"{}", headers={"Content-Type": "application/json"})
        resp = json.loads(urlopen(req, timeout=10).read())
        session_id = resp["session_id"]
    except Exception as e:
        return "error", f"SwarmUI session: {e}"

    gen_url = f"{args.swarm_url}/API/GenerateText2Image"
    payload = json.dumps({
        "session_id": session_id,
        "images": 1,
        "prompt": prompt,
        "negativeprompt": "",
        "model": args.model,
        "width": args.width,
        "height": args.height,
        "steps": args.steps,
        "cfgscale": 7.0,
        "seed": -1,
    }).encode()
    try:
        req = Request(gen_url, data=payload, headers={"Content-Type": "application/json"})
        resp = json.loads(urlopen(req, timeout=280).read())
        if "error" in resp or not resp.get("images"):
            return "blocked", str(resp.get("error", "no images"))
        img_b64 = resp["images"][0].split(",")[-1]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(img_b64))
        return "generated", ""
    except Exception as e:
        return "error", str(e)


def generate_one(provider: str, model: str, prompt: str,
                 output_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    if output_path.exists() and not args.overwrite:
        return "skipped", "already exists"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if provider == "sdxl":
        return call_swarm(prompt, output_path, args)
    else:
        return call_chatgpt_py(provider, model, prompt, output_path, args)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider", choices=["sdxl", "vertex", "gemini-vertex", "openai"],
                        default="sdxl")
    parser.add_argument("--model", default=None,
                        help="Override model name (provider-specific default used otherwise)")
    parser.add_argument("--prompt-set", choices=["source", "followup", "both"], default="both")
    parser.add_argument("--mr-ids", nargs="+", default=MR_IDS,
                        help="Which MR follow-up sets to generate")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-root", default=str(ROOT / "data" / "generated"))
    parser.add_argument("--python-exe", default="python3")
    parser.add_argument("--swarm-url", default="http://127.0.0.1:7801")
    parser.add_argument("--aspect-ratio", default="1:1")
    parser.add_argument("--quality", default="medium", help="OpenAI image quality (low/medium/high)")
    parser.add_argument("--max-attempts", type=int, default=3, dest="max_attempts",
                        help="Max attempts per prompt before giving up (set to 1 to disable retries)")
    parser.add_argument("--min-interval", type=float, default=0, dest="min_interval",
                        help="Minimum seconds between consecutive API calls (rate limiting, e.g. 13 for ~5 RPM)")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--project-id", default=None)
    parser.add_argument("--location", default="us-central1")
    parser.add_argument("--credentials-file", default=None)
    args = parser.parse_args()

    # Default models
    default_models = {
        "sdxl": "OfficialStableDiffusion/sd_xl_base_1.0.safetensors",
        "vertex": "imagen-3.0-generate-002",
        "gemini-vertex": "gemini-2.5-flash-image",
        "openai": "gpt-image-1",
    }
    model = args.model or default_models[args.provider]
    model_tag = f"{args.provider}_{model.split('/')[-1].replace('.', '_')[:30]}"
    out_dir = Path(args.output_root) / model_tag
    manifest_path = out_dir / "manifest.jsonl"

    source_prompts = load_source_prompts()
    metadata = load_metadata()

    jobs = []  # (idx, mr_id_or_none, prompt, output_path)

    # Source prompts
    if args.prompt_set in ("source", "both"):
        for i, p in enumerate(source_prompts):
            if i < args.start_index:
                continue
            if args.limit and len([j for j in jobs if j[1] is None]) >= args.limit:
                break
            if not p:
                continue
            fname = f"source_{i:04d}_{sanitize(p)}.png"
            jobs.append((i, None, p, out_dir / fname))

    # Follow-up prompts
    if args.prompt_set in ("followup", "both"):
        for mr_id in args.mr_ids:
            followups = load_followup_prompts(mr_id)
            if not followups:
                print(f"[warn] No followup file for MR-{mr_id}")
                continue
            for i, p in enumerate(followups):
                if i < args.start_index:
                    continue
                if not p:
                    continue
                fname = f"mr{mr_id}_{i:04d}_{sanitize(p)}.png"
                jobs.append((i, mr_id, p, out_dir / fname))

    print(f"Provider : {args.provider} / {model}")
    print(f"Output   : {out_dir}")
    print(f"Jobs     : {len(jobs)}")
    print()

    stats = {"generated": 0, "blocked": 0, "skipped": 0, "error": 0}
    for job_num, (idx, mr_id, prompt, img_path) in enumerate(jobs, 1):
        label = f"MR-{mr_id}" if mr_id else "source"
        meta = metadata.get(idx, {})
        ptype = meta.get("prompt_type", "unknown")
        print(f"[{job_num}/{len(jobs)}] {label} #{idx:03d} [{ptype}]  {prompt[:70]}")

        started = datetime.now(timezone.utc).isoformat()
        t0 = time.time()
        status, detail = generate_one(args.provider, model, prompt, img_path, args)
        elapsed = round(time.time() - t0, 2)

        stats[status] = stats.get(status, 0) + 1
        icon = {"generated": "✓", "blocked": "✗", "skipped": "-", "error": "!"}.get(status, "?")
        print(f"  {icon} {status}  ({elapsed}s){f'  [{detail[-150:]}]' if detail else ''}")

        append_manifest(manifest_path, {
            "prompt_index": idx,
            "mr_id": mr_id,
            "prompt_type": ptype,
            "category": meta.get("category", ""),
            "anchor_cue": meta.get("anchor_cue", ""),
            "prompt": prompt,
            "provider": args.provider,
            "model": model,
            "image_path": str(img_path),
            "status": status,
            "detail": detail,
            "started_at": started,
            "elapsed_s": elapsed,
        })
        if status not in ("skipped",):
            time.sleep(2.0)

    print(f"\nDone: {stats}")
    print(f"Manifest → {manifest_path}")


if __name__ == "__main__":
    main()
