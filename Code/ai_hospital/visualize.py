"""Overlay findings on the image so each report citation [Fi] maps to a visible region."""
import numpy as np
from PIL import Image, ImageDraw

PALETTE = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180)]


def draw_findings(image, findings):
    img = Image.fromarray(np.asarray(image)).convert("RGB")
    d = ImageDraw.Draw(img)
    for i, f in enumerate(findings):
        c = PALETTE[i % len(PALETTE)]
        d.rectangle(f.bbox, outline=c, width=3)
        tag = f"{f.id} {f.label} {f.confidence:.2f}"
        x, y = f.bbox[0], max(f.bbox[1] - 14, 0)
        d.rectangle([x, y, x + 7 * len(tag), y + 13], fill=c)
        d.text((x + 2, y), tag, fill=(0, 0, 0))
    return img
