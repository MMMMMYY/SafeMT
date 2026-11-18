import argparse
import json
import re
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


DEFAULT_SWARM_URL = "http://127.0.0.1:7801"
DEFAULT_MODEL = "OfficialStableDiffusion/sd_xl_base_1.0.safetensors"
ROOT = Path(__file__).resolve().parents[2]
SOURCE_JSON = ROOT / "llmorph/data/prompt-mutation/source_inputs/source_inputs.json"
T4_JSON = ROOT / "llmorph/data/pilot-study/followup_inputs/followup_inputs__prompt_mutation__203.json"
OUTPUT_ROOT = ROOT / "pilot study/generated_images/sdxl_local"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-generate source and MR-T4 prompts with local SDXL via SwarmUI."
    )
    parser.add_argument(
        "--prompt-set",
        choices=["source", "t4", "both"],
        default="both",
        help="Which prompt set to generate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of prompt pairs to process.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based prompt index to start from.",
    )
    parser.add_argument(
        "--swarm-url",
        default=DEFAULT_SWARM_URL,
        help="Base URL of the local SwarmUI server.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="SwarmUI model name.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Generation width.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Generation height.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="Inference steps.",
    )
    parser.add_argument(
        "--cfgscale",
        type=float,
        default=7.5,
        help="CFG scale.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=1000,
        help="Base seed. Prompt index is added to this value.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate even if a local file already exists.",
    )
    return parser.parse_args()


def load_prompt_list(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    prompts = []
    for item in data:
        if isinstance(item, list) and item:
            first = item[0]
            if isinstance(first, str):
                prompts.append(first)
            elif isinstance(first, list) and first and isinstance(first[0], str):
                prompts.append(first[0])
            else:
                prompts.append("")
        else:
            prompts.append("")
    return prompts


def sanitize_filename(text: str, max_len: int = 80) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    if not text:
        text = "prompt"
    return text[:max_len].rstrip("_")


def get_session(base_url: str, timeout: int) -> str:
    data = post_json(f"{base_url}/API/GetNewSession", {}, timeout)
    return data["session_id"]


def post_json(url: str, payload: dict, timeout: int) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc
    except URLError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def generate_one(
    *,
    base_url: str,
    session_id: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    steps: int,
    cfgscale: float,
    seed: int,
    timeout: int,
) -> str:
    payload = {
        "session_id": session_id,
        "images": 1,
        "prompt": prompt,
        "negativeprompt": "",
        "model": model,
        "width": width,
        "height": height,
        "steps": steps,
        "cfgscale": cfgscale,
        "seed": seed,
    }
    data = post_json(f"{base_url}/API/GenerateText2Image", payload, timeout)
    if "error" in data:
        raise RuntimeError(data["error"])
    images = data.get("images") or []
    if not images:
        raise RuntimeError(f"No image returned for prompt: {prompt}")
    return images[0]


def download_image(base_url: str, swarm_path: str, local_path: Path, timeout: int) -> None:
    encoded_path = quote(swarm_path.lstrip("/"), safe="/")
    request = Request(f"{base_url}/{encoded_path}", method="GET")
    with urlopen(request, timeout=timeout) as response:
        content = response.read()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(content)


def append_manifest(manifest_path: Path, record: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    source_prompts = load_prompt_list(SOURCE_JSON)
    t4_prompts = load_prompt_list(T4_JSON)

    if len(source_prompts) != len(t4_prompts):
        raise RuntimeError("Source and T4 prompt counts do not match.")

    prompt_sets = []
    if args.prompt_set in {"source", "both"}:
        prompt_sets.append(("source", source_prompts))
    if args.prompt_set in {"t4", "both"}:
        prompt_sets.append(("mr_t4_context_injection", t4_prompts))

    total = len(source_prompts)
    end_index = total if args.limit is None else min(total, args.start_index + args.limit)
    session_id = get_session(args.swarm_url, args.timeout)

    print(f"SwarmUI session: {session_id}")
    print(f"Model: {args.model}")
    print(f"Processing prompt indices: {args.start_index}..{end_index - 1}")

    for prompt_index in range(args.start_index, end_index):
        seed = args.base_seed + prompt_index
        for set_name, prompts in prompt_sets:
            prompt = prompts[prompt_index]
            if not prompt:
                print(f"[skip] {set_name} #{prompt_index + 1:02d}: empty prompt")
                continue

            prompt_slug = sanitize_filename(prompt)
            local_dir = OUTPUT_ROOT / set_name
            local_path = local_dir / f"{prompt_index + 1:02d}_{prompt_slug}.png"
            manifest_path = local_dir / "manifest.jsonl"

            if local_path.exists() and not args.overwrite:
                print(f"[skip] {set_name} #{prompt_index + 1:02d}: {local_path.name}")
                continue

            print(f"[run]  {set_name} #{prompt_index + 1:02d} seed={seed}: {prompt}")
            started_at = time.time()
            swarm_path = generate_one(
                base_url=args.swarm_url,
                session_id=session_id,
                model=args.model,
                prompt=prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                cfgscale=args.cfgscale,
                seed=seed,
                timeout=args.timeout,
            )
            download_image(args.swarm_url, swarm_path, local_path, args.timeout)
            elapsed = round(time.time() - started_at, 2)

            append_manifest(
                manifest_path,
                {
                    "prompt_index": prompt_index + 1,
                    "prompt_set": set_name,
                    "prompt": prompt,
                    "seed": seed,
                    "model": args.model,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "cfgscale": args.cfgscale,
                    "swarm_path": swarm_path,
                    "local_path": str(local_path),
                    "elapsed_seconds": elapsed,
                },
            )
            print(f"[done] {set_name} #{prompt_index + 1:02d} -> {local_path}")


if __name__ == "__main__":
    main()
