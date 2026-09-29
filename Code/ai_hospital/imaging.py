"""Image loading (PNG/JPG or DICOM) to an 8-bit RGB array."""
from pathlib import Path

import numpy as np


def load_image(path):
    path = Path(path)
    if path.suffix.lower() in (".dcm", ""):
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        if arr.ndim == 3 and arr.shape[-1] not in (3, 4):  # volume: take the middle slice
            arr = arr[arr.shape[0] // 2]
        center, width = getattr(ds, "WindowCenter", None), getattr(ds, "WindowWidth", None)
        if center is not None and width is not None:
            c = float(center[0] if hasattr(center, "__len__") else center)
            w = float(width[0] if hasattr(width, "__len__") else width)
            arr = np.clip((arr - (c - w / 2)) / w, 0, 1)
        else:
            arr = (arr - arr.min()) / (np.ptp(arr) + 1e-8)
        arr = (arr * 255).astype(np.uint8)
    else:
        from PIL import Image

        arr = np.asarray(Image.open(path).convert("RGB"))
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    return arr


def describe_location(bbox, shape, anatomy_hint=""):
    """Coarse, modality-neutral location from the box centre (image left = patient right on radiographs)."""
    h, w = shape[:2]
    cx, cy = (bbox[0] + bbox[2]) / 2 / w, (bbox[1] + bbox[3]) / 2 / h
    vert = "upper" if cy < 1 / 3 else "lower" if cy > 2 / 3 else "middle"
    horiz = "image-left" if cx < 1 / 3 else "image-right" if cx > 2 / 3 else "central"
    where = f"{vert} {horiz}"
    return f"{where} {anatomy_hint}".strip()
