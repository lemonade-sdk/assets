#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


TARGET_HEIGHT = 64
REQUEST_TIMEOUT = 30


@dataclass
class EngineImage:
    alt: str
    src: str


ENGINE_IMAGES = [
    EngineImage("llama.cpp", "https://user-images.githubusercontent.com/1991296/230134379-7181e485-c521-4d23-a0d6-f7b3b61ba524.png"),
    EngineImage("ONNX Runtime", "https://raw.githubusercontent.com/microsoft/onnxruntime/main/docs/images/ONNX_Runtime_logo.png"),
    EngineImage("FastFlowLM", "https://raw.githubusercontent.com/FastFlowLM/FastFlowLM/main/assets/logo_next_to_flm.png"),
    EngineImage("Ryzen AI SW", "https://www.phoronix.net/image.php?id=2025&image=ryzen_ai_sw_1"),
    EngineImage("ROCm", "https://upload.wikimedia.org/wikipedia/commons/0/06/20467978-A_AMD_ROCm_Lockup_85tall.png"),
    EngineImage("Hugging Face", "https://huggingface.co/datasets/huggingface/brand-assets/resolve/main/hf-logo.png"),
    EngineImage("Vulkan", "https://upload.wikimedia.org/wikipedia/commons/thumb/f/f8/Vulkan_API_logo.svg/330px-Vulkan_API_logo.svg.png"),
    EngineImage("whisper.cpp", "https://user-images.githubusercontent.com/1991296/235238348-05d0f6a4-da44-4900-a1de-d0707e75b763.jpeg"),
    EngineImage("stable-diffusion.cpp", "https://raw.githubusercontent.com/leejet/stable-diffusion.cpp/master/assets/logo.png"),
    EngineImage("Kokoros", "https://camo.githubusercontent.com/8b7b5be44f4ee1542bbcf55b2ab7e5e51f5e020f6b497bfab3e596a3b4715b75/68747470733a2f2f696d67323032332e636e626c6f67732e636f6d2f626c6f672f333537323332332f3230323530312f333537323332332d32303235303131323138343130303337382d3930373938383637302e6a7067"),
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


def download_image(session: requests.Session, url: str) -> Image.Image:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def crop_transparent_edges(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value > 0 else 0).getbbox()
    return rgba.crop(bbox) if bbox else rgba


def normalize_height(image: Image.Image, target_height: int = TARGET_HEIGHT) -> Image.Image:
    trimmed = crop_transparent_edges(image)
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

    for engine_image in ENGINE_IMAGES:
        normalized = normalize_height(download_image(session, engine_image.src))
        output_path = output_dir / f"{slugify(engine_image.alt)}.png"
        normalized.save(output_path, format="PNG")
        print(f"{engine_image.alt}: {output_path} <- {engine_image.src}")

    print(f"Saved normalized icons to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
