"""Detector (YOLO) and segmenter (U-Net) wrappers that emit `Finding`s."""
from pathlib import Path

import numpy as np

from .imaging import describe_location
from .schema import Finding


class YOLODetector:
    def __init__(self, weights, conf=0.25):
        from ultralytics import YOLO

        if not Path(weights).exists():
            raise FileNotFoundError(f"detector weights not found: {weights}")
        self.model = YOLO(weights)
        self.conf = conf

    def __call__(self, image, anatomy_hint=""):
        res = self.model.predict(image, conf=self.conf, verbose=False)[0]
        findings = []
        for i, (box, c, k) in enumerate(zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())):
            findings.append(Finding(id=f"F{i + 1}", label=res.names[int(k)], confidence=float(c), bbox=tuple(box),
                                    location=describe_location(box, image.shape, anatomy_hint)))
        return findings


class UNetSegmenter:
    """Refines each finding with a mask inside its box (TorchScript U-Net, 1-channel sigmoid output)."""

    def __init__(self, weights, size=256, threshold=0.5):
        import torch

        if not Path(weights).exists():
            raise FileNotFoundError(f"segmenter weights not found: {weights}")
        self.torch = torch
        self.model = torch.jit.load(weights, map_location="cpu").eval()
        self.size, self.threshold = size, threshold

    def __call__(self, image, findings):
        torch = self.torch
        for f in findings:
            x1, y1, x2, y2 = (int(round(v)) for v in f.bbox)
            crop = image[max(y1, 0):y2, max(x1, 0):x2].mean(-1)
            if crop.size == 0:
                continue
            t = torch.tensor(crop, dtype=torch.float32)[None, None] / 255.0
            t = torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear")
            with torch.no_grad():
                mask = torch.sigmoid(self.model(t))[0, 0] > self.threshold
            f.area_px = int(mask.float().mean().item() * crop.shape[0] * crop.shape[1])
        return findings
