#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageChops
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TARGET_HEIGHT = 64
SVG_RENDER_HEIGHT = 512
REQUEST_TIMEOUT = 30
WHITE_TRIM_TOLERANCE = 16


@dataclass
class EngineImage:
    alt: str
    src: str


# One entry per engine in the lemonade "Supported Configurations" table
# (https://github.com/lemonade-sdk/lemonade#supported-configurations), plus
# shared platform logos (ROCm, Vulkan, Hugging Face, Ryzen AI SW).
ENGINE_IMAGES = [
    # llamacpp — official brand kit at https://github.com/ggml-org/llama.brand
    EngineImage("llama.cpp", "https://raw.githubusercontent.com/ggml-org/llama.brand/master/logo/llama-cpp/logo-llama-cpp-light.svg"),
    # onnxruntime
    EngineImage("ONNX Runtime", "https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/images/ONNX_Runtime_logo.png"),
    # flm
    EngineImage("FastFlowLM", "https://raw.githubusercontent.com/FastFlowLM/FastFlowLM/main/assets/logo_next_to_flm.png"),
    # ryzenai-llm
    EngineImage("Ryzen AI SW", "https://www.phoronix.net/image.php?id=2025&image=ryzen_ai_sw_1"),
    EngineImage("ROCm", "https://upload.wikimedia.org/wikipedia/commons/0/06/20467978-A_AMD_ROCm_Lockup_85tall.png"),
    EngineImage("Hugging Face", "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"),
    EngineImage("Vulkan", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Vulkan_API_logo.svg/330px-Vulkan_API_logo.svg.png"),
    # whispercpp
    EngineImage("whisper.cpp", "https://user-images.githubusercontent.com/1991296/235238348-05d0f6a4-da44-4900-a1de-d0707e75b763.jpeg"),
    # sd-cpp
    EngineImage("stable-diffusion.cpp", "https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/assets/logo.png"),
    # kokoro — lemonade wraps the Kokoros backend (https://github.com/lucasjinreal/Kokoros)
    EngineImage("Kokoros", "https://camo.githubusercontent.com/8b7b5be44f4ee1542bbcf55b2ab7e5e51f5e020f6b497bfab3e596a3b4715b75/68747470733a2f2f696d67323032332e636e626c6f67732e636f6d2f626c6f672f333537323332332f3230323530312f333537323332332d32303235303131323138343130303337382d3930373938383637302e6a7067"),
    # vllm
    EngineImage("vLLM", "https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png"),
    # moonshine
    EngineImage("Moonshine", "https://github.com/moonshine-ai/moonshine/raw/main/images/logo.png"),
    # openmoss — no dedicated MOSS-TTSD logo; official OpenMOSS wordmark from the repo
    EngineImage("OpenMOSS", "https://raw.githubusercontent.com/OpenMOSS/MOSS-TTSD/main/asset/OpenMOSS_Logo.svg"),
    # trellis — TRELLIS.2 has no logo asset; official TRELLIS wordmark (animated webp, first frame)
    EngineImage("TRELLIS", "https://github.com/microsoft/TRELLIS/raw/main/assets/logo.webp"),
    # acestep — no standalone project logo in the repo; Hugging Face org avatar
    EngineImage("ACE-Step", "https://cdn-avatars.huggingface.co/v1/production/uploads/6209bb6ede1c3ff3ec37620c/xk4TNYgu3UPz74tAgzTrA.jpeg"),
    # thinksound — intentionally omitted: the project has no logo of its own, and the
    # publisher org marks (FunAudioLLM/QwenAudio) are just Qwen-family branding.
]


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "image"


def build_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session = requests.Session()
    session.headers["User-Agent"] = "lsdk-engine-icon-fetcher/1.0"
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def is_svg(url: str, content: bytes) -> bool:
    if url.split("?")[0].lower().endswith(".svg"):
        return True
    head = content[:512].lstrip().lower()
    return head.startswith(b"<svg") or (head.startswith(b"<?xml") and b"<svg" in head)


def download_image(session: requests.Session, url: str) -> Image.Image:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content = response.content

    if is_svg(url, content):
        import cairosvg

        content = cairosvg.svg2png(bytestring=content, output_height=SVG_RENDER_HEIGHT)

    image = Image.open(BytesIO(content))
    if getattr(image, "is_animated", False):
        image.seek(0)
    return image.convert("RGBA")


def crop_padding(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")

    alpha = rgba.getchannel("A")
    lo, hi = alpha.getextrema()
    if lo < 255:
        bbox = alpha.point(lambda value: 255 if value > 0 else 0).getbbox()
        return rgba.crop(bbox) if bbox else rgba

    # Fully opaque image (e.g. jpeg): trim uniform border color instead.
    border_color = rgba.getpixel((0, 0))
    background = Image.new("RGBA", rgba.size, border_color)
    diff = ImageChops.difference(rgba, background).convert("L")
    bbox = diff.point(lambda value: 255 if value > WHITE_TRIM_TOLERANCE else 0).getbbox()
    return rgba.crop(bbox) if bbox else rgba


def normalize_height(image: Image.Image, target_height: int = TARGET_HEIGHT) -> Image.Image:
    trimmed = crop_padding(image)
    if trimmed.height == 0:
        raise RuntimeError("Image became empty after trimming")

    scale = target_height / trimmed.height
    resized = trimmed.resize(
        (max(1, round(trimmed.width * scale)), target_height),
        Image.Resampling.LANCZOS,
    )
    return resized


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    print(f"Preparing {len(ENGINE_IMAGES)} engine icons")

    failures = []
    for engine_image in ENGINE_IMAGES:
        try:
            normalized = normalize_height(download_image(session, engine_image.src))
        except Exception as error:
            failures.append((engine_image.alt, error))
            print(f"{engine_image.alt}: FAILED ({error})")
            continue
        output_path = output_dir / f"{slugify(engine_image.alt)}.png"
        normalized.save(output_path, format="PNG")
        print(f"{engine_image.alt}: {output_path} <- {engine_image.src}")

    print(f"Saved normalized icons to {output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
