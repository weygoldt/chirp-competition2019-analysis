"""Composite the transparent README figures onto GitHub's page backgrounds."""

import sys
from pathlib import Path

from PIL import Image

BG = {
    "light": (255, 255, 255, 255),
    "dark": (13, 17, 23, 255),
}  # GitHub's light and dark page

for name in sys.argv[1:]:
    path = Path(name)
    theme = "dark" if path.stem.endswith("_dark") else "light"
    img = Image.open(path).convert("RGBA")
    page = Image.new("RGBA", img.size, BG[theme])
    page.alpha_composite(img)
    out = path.with_name(path.stem + "_preview.png")
    page.convert("RGB").save(out)
    print(out, img.size, f"{path.stat().st_size / 1e3:.0f} kB")
