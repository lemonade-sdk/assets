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


SIZE = 64
REQUEST_TIMEOUT = 30


@dataclass
class PlatformImage:
    alt: str
    src: str


PLATFORM_IMAGES = [
    PlatformImage("Ubuntu", "https://upload.wikimedia.org/wikipedia/commons/thumb/9/9e/UbuntuCoF.svg/960px-UbuntuCoF.svg.png"),
    PlatformImage("Snapcraft", "https://commons.wikimedia.org/wiki/Special:FilePath/Snapcraft-logo-bird.svg?width=512"),
    PlatformImage("Arch Linux", "https://upload.wikimedia.org/wikipedia/commons/thumb/1/13/Arch_Linux_%22Crystal%22_icon.svg/330px-Arch_Linux_%22Crystal%22_icon.svg.png"),
    PlatformImage("Fedora", "https://upload.wikimedia.org/wikipedia/commons/thumb/4/41/Fedora_icon_%282021%29.svg/960px-Fedora_icon_%282021%29.svg.png"),
    PlatformImage("Debian", "https://www.clipartmax.com/png/middle/179-1793137_debian-logo-debian-logo-svg.png"),
    PlatformImage("Docker", "https://commons.wikimedia.org/wiki/Special:FilePath/Docker_(container_engine)_logo_(cropped).png"),
    PlatformImage("Windows", "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/Windows_logo_-_2012.png/500px-Windows_logo_-_2012.png"),
    PlatformImage("Apple", "https://cdn-icons-png.flaticon.com/512/2/2235.png"),
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
    session.headers["User-Agent"] = "lsdk-platform-icon-fetcher/1.0"
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def crop_transparent_edges(image: Image.Image) -> Image.Image:
    alpha = image.getchannel("A")
    bbox = alpha.point(lambda value: 255 if value > 0 else 0).getbbox()
    return image.crop(bbox) if bbox else image


def corners_are_similar(image: Image.Image, tolerance: int = 18) -> bool:
    w, h = image.size
    coords = [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]
    corners = [image.getpixel(coord) for coord in coords]

    if any(pixel[3] < 250 for pixel in corners):
        return False

    base = corners[0]
    for pixel in corners[1:]:
        if max(abs(base[i] - pixel[i]) for i in range(3)) > tolerance:
            return False
    return True


def crop_flat_border(image: Image.Image, tolerance: int = 18) -> Image.Image:
    if image.width < 4 or image.height < 4 or not corners_are_similar(image, tolerance=tolerance):
        return image

    bg = image.getpixel((0, 0))
    xs: list[int] = []
    ys: list[int] = []
    pixels = image.load()

    for y in range(image.height):
        for x in range(image.width):
            pixel = pixels[x, y]
            if pixel[3] == 0:
                continue
            if max(abs(pixel[i] - bg[i]) for i in range(3)) > tolerance or abs(pixel[3] - bg[3]) > tolerance:
                xs.append(x)
                ys.append(y)

    if not xs or not ys:
        return image

    bbox = (min(xs), min(ys), max(xs) + 1, max(ys) + 1)
    return image.crop(bbox)


def normalize_icon(image: Image.Image, size: int = SIZE) -> Image.Image:
    rgba = image.convert("RGBA")
    trimmed = crop_flat_border(crop_transparent_edges(rgba))
    if trimmed.width == 0 or trimmed.height == 0:
        raise RuntimeError("Image became empty after trimming")

    scale = min(size / trimmed.width, size / trimmed.height)
    resized = trimmed.resize(
        (max(1, round(trimmed.width * scale)), max(1, round(trimmed.height * scale))),
        Image.Resampling.LANCZOS,
    )

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    x = (size - resized.width) // 2
    y = (size - resized.height) // 2
    canvas.paste(resized, (x, y), resized)
    return canvas


def download_image(session: requests.Session, url: str) -> Image.Image:
    response = session.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return Image.open(BytesIO(response.content))


def main() -> int:
    output_dir = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)

    session = build_session()
    print(f"Preparing {len(PLATFORM_IMAGES)} platform icons")

    for platform_image in PLATFORM_IMAGES:
        normalized = normalize_icon(download_image(session, platform_image.src))
        output_path = output_dir / f"{slugify(platform_image.alt)}.png"
        normalized.save(output_path, format="PNG")
        print(f"{platform_image.alt}: {output_path} <- {platform_image.src}")

    print(f"Saved normalized icons to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
