import argparse
import base64
import json
import mimetypes
import os
import subprocess
import time
from pathlib import Path

import requests
from google.auth.transport.requests import Request
from google.oauth2 import service_account


DEFAULT_PROMPT = (
    "Create a polished cinematic poster inspired by the provided reference image. "
    "Keep the composition minimal and dramatic, with deep black negative space, "
    "a premium editorial aesthetic, subtle volumetric lighting, crisp edges, "
    "rich texture detail, high contrast, and realistic rendering quality."
)
DEFAULT_REFERENCE = "a2800fa4-4d99-4caf-ac7f-ae34d6238832.png"
OPENAI_KEY_FILE = "openAI_key.txt"
HF_KEY_FILE = "hf_key.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate one image with OpenAI, Gemini, Vertex, FLUX, or SDXL.")
    parser.add_argument(
        "--provider",
        choices=["gemini-vertex", "openai", "vertex", "flux", "sdxl"],
        default="gemini-vertex",
        help="API provider to use.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used for image generation.",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help="Optional reference image path. If missing, a text-only generation is used.",
    )
    parser.add_argument(
        "--output",
        default="generated_reference_variation.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--size",
        default="1024x1024",
        help="OpenAI image size. Ignored by Gemini.",
    )
    parser.add_argument(
        "--quality",
        default="high",
        help="OpenAI image quality. Ignored by Gemini.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="2:3",
        help="Gemini aspect ratio.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the provider default model.",
    )
    parser.add_argument(
        "--key-file",
        default=None,
        help="Optional path to a text file containing the API key.",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        help="Google Cloud project id for Vertex AI.",
    )
    parser.add_argument(
        "--location",
        default=os.environ.get("VERTEX_LOCATION", "us-central1"),
        help="Vertex AI location.",
    )
    parser.add_argument(
        "--credentials-file",
        default=os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"),
        help="Path to a Google service account JSON file for Vertex AI.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional generation seed for providers that support it.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=4,
        help="Inference steps for FLUX via Hugging Face.",
    )
    parser.add_argument(
        "--guidance-scale",
        type=float,
        default=3.5,
        help="Guidance scale for FLUX via Hugging Face.",
    )
    parser.add_argument(
        "--hf-provider",
        default=None,
        help="Optional Hugging Face inference provider override, for example hf-inference or fal-ai.",
    )
    return parser.parse_args()


def read_api_key(provider: str, key_file: str | None) -> str:
    if provider in ("vertex", "gemini-vertex"):
        return ""

    env_var = {
        "openai": "OPENAI_API_KEY",
        "flux": "HF_TOKEN",
        "sdxl": "HF_TOKEN",
    }[provider]
    key = os.environ.get(env_var)
    if key:
        return key.strip()

    fallback_file = {
        "openai": OPENAI_KEY_FILE,
        "flux": HF_KEY_FILE,
        "sdxl": HF_KEY_FILE,
    }[provider]
    key_path = Path(key_file or fallback_file)
    if key_path.exists():
        return key_path.read_text().strip()

    raise RuntimeError(f"{env_var} is not set and key file {key_path} was not found.")


def detect_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "application/octet-stream"


def save_base64_image(image_base64: str, output_path: Path) -> None:
    output_path.write_bytes(base64.b64decode(image_base64))


def parse_retry_delay(error_payload: dict) -> float | None:
    details = error_payload.get("error", {}).get("details", [])
    for detail in details:
        retry_delay = detail.get("retryDelay")
        if not retry_delay:
            continue
        if retry_delay.endswith("s"):
            try:
                return float(retry_delay[:-1])
            except ValueError:
                return None
    return None


def parse_size(size: str) -> tuple[int, int]:
    try:
        width_str, height_str = size.lower().split("x", maxsplit=1)
        return int(width_str), int(height_str)
    except ValueError as exc:
        raise RuntimeError(f"Invalid size format: {size}. Expected WIDTHxHEIGHT.") from exc


def get_vertex_access_token(credentials_file: str | None) -> str:
    if not credentials_file:
        gcloud_candidates = [
            "/Applications/google-cloud-sdk/bin/gcloud",
            "gcloud",
        ]
        for gcloud_path in gcloud_candidates:
            try:
                result = subprocess.run(
                    [gcloud_path, "auth", "application-default", "print-access-token"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                continue

            token = result.stdout.strip()
            if token:
                return token

        raise RuntimeError(
            "Vertex AI requires --credentials-file / GOOGLE_APPLICATION_CREDENTIALS, "
            "or a working gcloud ADC login."
        )

    credentials = service_account.Credentials.from_service_account_file(
        credentials_file,
        scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
    credentials.refresh(Request())
    token = credentials.token
    if not token:
        raise RuntimeError("Failed to obtain a Vertex AI access token.")
    return token


def generate_with_openai(
    prompt: str,
    output_path: Path,
    size: str,
    quality: str,
    reference_path: Path | None,
    api_key: str,
    model: str | None,
) -> None:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    model_name = model or "gpt-image-1"

    if reference_path and reference_path.exists():
        with reference_path.open("rb") as image_file:
            result = client.images.edit(
                model=model_name,
                image=image_file,
                prompt=prompt,
                size=size,
                quality=quality,
            )
        image_base64 = result.data[0].b64_json
    else:
        response = requests.post(
            "https://api.openai.com/v1/images/generations",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model_name,
                "prompt": prompt,
                "size": size,
                "quality": quality,
            },
            timeout=300,
        )
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(response.text) from exc
        result = response.json()
        image_base64 = result["data"][0]["b64_json"]

    save_base64_image(image_base64, output_path)


def generate_with_gemini_vertex(
    prompt: str,
    output_path: Path,
    reference_path: Path | None,
    aspect_ratio: str,
    project_id: str | None,
    location: str,
    credentials_file: str | None,
    model: str | None,
) -> None:
    """Gemini image generation routed through Vertex AI (project billing, no API key, no free-tier cap)."""
    if not project_id:
        raise RuntimeError("Vertex AI requires --project-id or GOOGLE_CLOUD_PROJECT.")

    access_token = get_vertex_access_token(credentials_file)
    model_name = model or "gemini-2.5-flash-image"

    parts = [{"text": prompt}]
    if reference_path and reference_path.exists():
        image_bytes = reference_path.read_bytes()
        parts.append(
            {
                "inline_data": {
                    "mime_type": detect_mime_type(reference_path),
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            }
        )

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "imageConfig": {
                "aspectRatio": aspect_ratio,
            },
        },
    }

    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model_name}:generateContent"
    )

    result = None
    for attempt in range(2):
        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=300,
        )
        if response.ok:
            result = response.json()
            break

        try:
            error_payload = response.json()
        except ValueError:
            error_payload = {"raw": response.text}

        if response.status_code == 429 and attempt == 0:
            retry_after = parse_retry_delay(error_payload) or 30
            time.sleep(retry_after)
            continue

        raise RuntimeError(str(error_payload))

    if result is None:
        raise RuntimeError("Gemini (Vertex) request failed without a response payload.")

    candidates = result.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline_data = part.get("inlineData")
            if inline_data and inline_data.get("data"):
                save_base64_image(inline_data["data"], output_path)
                return

    raise RuntimeError(f"No image was returned by Gemini (Vertex). Response: {result}")


def generate_with_vertex(
    prompt: str,
    output_path: Path,
    aspect_ratio: str,
    project_id: str | None,
    location: str,
    credentials_file: str | None,
    model: str | None,
) -> None:
    if not project_id:
        raise RuntimeError("Vertex AI requires --project-id or GOOGLE_CLOUD_PROJECT.")

    access_token = get_vertex_access_token(credentials_file)
    model_name = model or "imagen-3.0-generate-002"
    url = (
        f"https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/"
        f"locations/{location}/publishers/google/models/{model_name}:predict"
    )
    payload = {
        "instances": [{"prompt": prompt}],
        "parameters": {
            "sampleCount": 1,
            "aspectRatio": aspect_ratio,
            "safetyFilterLevel": "block_some",
            "personGeneration": "allow_adult",
        },
    }

    response = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=300,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(response.text) from exc

    result = response.json()
    predictions = result.get("predictions", [])
    if not predictions:
        raise RuntimeError(f"Vertex AI blocked the request (no predictions returned). Response: {json.dumps(result)}")

    image_base64 = (
        predictions[0].get("bytesBase64Encoded")
        or predictions[0].get("image", {}).get("bytesBase64Encoded")
    )
    if not image_base64:
        raise RuntimeError(f"Vertex AI returned no image bytes. Response: {json.dumps(result)}")

    save_base64_image(image_base64, output_path)


def generate_with_flux(
    prompt: str,
    output_path: Path,
    size: str,
    api_key: str,
    model: str | None,
    provider: str | None,
    seed: int | None,
    steps: int,
    guidance_scale: float,
) -> None:
    from huggingface_hub import InferenceClient

    model_name = model or "black-forest-labs/FLUX.1-schnell"
    width, height = parse_size(size)
    client = InferenceClient(api_key=api_key, provider=provider)
    image = client.text_to_image(
        prompt,
        model=model_name,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        seed=seed,
    )
    image.save(output_path)


def resolve_hf_model(provider: str, requested_model: str | None) -> str:
    if requested_model:
        aliases = {
            "flux1": "black-forest-labs/FLUX.1-dev",
            "flux1.0": "black-forest-labs/FLUX.1-dev",
            "flux2": "black-forest-labs/FLUX.2-dev",
            "flux2.0": "black-forest-labs/FLUX.2-dev",
            "flux-dev": "black-forest-labs/FLUX.1-dev",
            "flux-fast": "black-forest-labs/FLUX.1-schnell",
            "sdxl": "stabilityai/stable-diffusion-xl-base-1.0",
        }
        return aliases.get(requested_model.lower(), requested_model)

    if provider == "flux":
        return "black-forest-labs/FLUX.1-dev"
    if provider == "sdxl":
        return "stabilityai/stable-diffusion-xl-base-1.0"
    raise RuntimeError(f"No default Hugging Face model configured for provider {provider}.")


def resolve_hf_provider(provider: str, model_name: str) -> str | None:
    if model_name.startswith("black-forest-labs/FLUX.1") or model_name.startswith("black-forest-labs/FLUX.2"):
        return "fal-ai"
    return None


def main() -> None:
    args = parse_args()
    reference_path = Path(args.reference) if args.reference else None
    output_path = Path(args.output)
    api_key = read_api_key(args.provider, args.key_file)

    if args.provider == "gemini-vertex":
        generate_with_gemini_vertex(
            prompt=args.prompt,
            output_path=output_path,
            reference_path=reference_path,
            aspect_ratio=args.aspect_ratio,
            project_id=args.project_id,
            location=args.location,
            credentials_file=args.credentials_file,
            model=args.model,
        )
    elif args.provider == "vertex":
        generate_with_vertex(
            prompt=args.prompt,
            output_path=output_path,
            aspect_ratio=args.aspect_ratio,
            project_id=args.project_id,
            location=args.location,
            credentials_file=args.credentials_file,
            model=args.model,
        )
    elif args.provider == "flux":
        flux_model = resolve_hf_model("flux", args.model)
        generate_with_flux(
            prompt=args.prompt,
            output_path=output_path,
            size=args.size,
            api_key=api_key,
            model=flux_model,
            provider=args.hf_provider or resolve_hf_provider("flux", flux_model),
            seed=args.seed,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
        )
    elif args.provider == "sdxl":
        sdxl_model = resolve_hf_model("sdxl", args.model)
        generate_with_flux(
            prompt=args.prompt,
            output_path=output_path,
            size=args.size,
            api_key=api_key,
            model=sdxl_model,
            provider=args.hf_provider or resolve_hf_provider("sdxl", sdxl_model),
            seed=args.seed,
            steps=max(args.steps, 20),
            guidance_scale=max(args.guidance_scale, 7.0),
        )
    else:
        generate_with_openai(
            prompt=args.prompt,
            output_path=output_path,
            size=args.size,
            quality=args.quality,
            reference_path=reference_path,
            api_key=api_key,
            model=args.model,
        )

    print(f"Provider: {args.provider}")
    print(f"Saved image to {output_path.resolve()}")
    print(f"Prompt used: {args.prompt}")
    if reference_path and reference_path.exists():
        print(f"Reference image: {reference_path.resolve()}")
    else:
        print("Reference image: none")


if __name__ == "__main__":
    main()
