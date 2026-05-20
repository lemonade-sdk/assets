#!/usr/bin/env python3
"""Generate Lemonade logo PNGs for the brand kit.

The lockup mirrors the docs/site navbar: Lemonade icon, a 0.25em gap, and the
Plus Jakarta Sans ExtraBold wordmark. Outputs have transparent backgrounds.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
SOURCE_ICON = ROOT / "source" / "lemonade-homepage-favicon.ico"
FONT_PATH = ROOT / "vendor" / "PlusJakartaSans-ExtraBold.ttf"

WORDMARK = "Lemonade"
TEXT_COLORS = {
    "light": "#18181B",
    "dark": "#F2EFE5",
}

# Output heights in pixels. Width is calculated from the lockup content plus
# proportional padding so the logo is not cramped in social cards or docs.
HEIGHTS = (128, 256, 512, 1024)


def draw_tracked_text(
    image: Image.Image,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont,
    fill: str,
    tracking: float,
) -> None:
    draw = ImageDraw.Draw(image)
    x, y = xy
    for char in text:
        draw.text((round(x), round(y)), char, font=font, fill=fill)
        x += draw.textlength(char, font=font) + tracking


def tracked_text_width(text: str, font: ImageFont.FreeTypeFont, tracking: float) -> float:
    draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    width = sum(draw.textlength(char, font=font) for char in text)
    return width + tracking * max(len(text) - 1, 0)


def render_logo(mode: str, height: int) -> Path:
    content_height = round(height * 0.70)
    pad_y = round((height - content_height) / 2)
    pad_x = round(height * 0.18)

    icon_size = content_height
    font_size = round(icon_size * (26.25 / 31.5))
    gap = round(icon_size * 0.25)
    tracking = -0.04 * font_size

    font = ImageFont.truetype(str(FONT_PATH), font_size)
    text_bbox = font.getbbox(WORDMARK)
    text_width = tracked_text_width(WORDMARK, font, tracking)
    width = round(pad_x + icon_size + gap + text_width + pad_x)

    canvas = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    icon = Image.open(SOURCE_ICON).convert("RGBA").resize(
        (icon_size, icon_size), Image.Resampling.LANCZOS
    )

    icon_x = pad_x
    icon_y = pad_y
    canvas.alpha_composite(icon, (icon_x, icon_y))

    text_x = icon_x + icon_size + gap
    text_y = (height - (text_bbox[3] - text_bbox[1])) / 2 - text_bbox[1]
    draw_tracked_text(canvas, (text_x, text_y), WORDMARK, font, TEXT_COLORS[mode], tracking)

    out = ROOT / f"lemonade-logo-{mode}-{height}h.png"
    canvas.save(out, "PNG", optimize=True)
    return out


def main() -> None:
    missing = [path for path in (SOURCE_ICON, FONT_PATH) if not path.exists()]
    if missing:
        names = ", ".join(str(path.relative_to(ROOT)) for path in missing)
        raise SystemExit(f"Missing required source file(s): {names}")

    for mode in TEXT_COLORS:
        for height in HEIGHTS:
            out = render_logo(mode, height)
            print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
