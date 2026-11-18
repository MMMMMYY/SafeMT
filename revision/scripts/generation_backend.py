"""Provider-call helpers used by run_repeated_generation.py.

Tables V--VII: collection source. Prompt inputs are read through package_paths;
the historical provider client is included as provider_client.py. Analysis from
frozen records does not invoke this module.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from package_paths import ROOT, GENERATED, load_source_prompts, load_metadata, load_followups as load_followup_prompts
PROVIDER_CLIENT = Path(__file__).with_name('provider_client.py')
MR_IDS = ["51", "8", "204", "155", "artistic_medium", "photographic_style", "environment_details", "camera_perspective"]


# ── helpers ───────────────────────────────────────────────────────────────────

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


def call_provider_client(provider: str, model: str, prompt: str,
                    output_path: Path, args: argparse.Namespace) -> tuple[str, str]:
    """Call the provider client. Returns (status, detail)."""
    cmd = [
        args.python_exe, str(PROVIDER_CLIENT),
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
        return call_provider_client(provider, model, prompt, output_path, args)


# ── main ──────────────────────────────────────────────────────────────────────
