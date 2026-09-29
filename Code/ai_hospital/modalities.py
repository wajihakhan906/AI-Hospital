"""Supported modalities and how to recognise them from DICOM headers."""
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ModalityConfig:
    key: str
    name: str
    detector_weights: str
    segmenter_weights: Optional[str] = None
    anatomy_hint: str = ""


MODALITIES = {
    "xray": ModalityConfig("xray", "Chest X-ray", "weights/xray_yolo.pt", "weights/xray_unet.pt", "chest"),
    "cbct": ModalityConfig("cbct", "Dental CBCT", "weights/cbct_yolo.pt", "weights/cbct_unet.pt", "maxillofacial"),
    "ct": ModalityConfig("ct", "Computed Tomography", "weights/ct_yolo.pt", "weights/ct_unet.pt"),
    "mri": ModalityConfig("mri", "Magnetic Resonance Imaging", "weights/mri_yolo.pt", "weights/mri_unet.pt", "brain"),
    "us": ModalityConfig("us", "Ultrasound", "weights/us_yolo.pt", "weights/us_unet.pt"),
    "petct": ModalityConfig("petct", "PET-CT", "weights/petct_yolo.pt"),
    "mammography": ModalityConfig("mammography", "Mammography", "weights/mammo_yolo.pt", "weights/mammo_unet.pt", "breast"),
    "angiography": ModalityConfig("angiography", "Angiography", "weights/angio_yolo.pt", "weights/angio_unet.pt", "vascular"),
}

# DICOM (0008,0060) Modality codes -> our keys
DICOM_CODES = {"CR": "xray", "DX": "xray", "CT": "ct", "MR": "mri", "US": "us", "PT": "petct",
               "MG": "mammography", "XA": "angiography", "RF": "angiography", "IO": "xray", "PX": "xray"}


def detect_modality(path, override=None):
    """User override > DICOM header > filename keyword."""
    if override:
        if override not in MODALITIES:
            raise ValueError(f"unknown modality {override!r}; choose from {sorted(MODALITIES)}")
        return override
    path = Path(path)
    if path.suffix.lower() in (".dcm", ""):
        try:
            import pydicom

            ds = pydicom.dcmread(str(path), stop_before_pixels=True)
            code = getattr(ds, "Modality", "")
            if code == "CT" and "CBCT" in str(getattr(ds, "SeriesDescription", "")).upper():
                return "cbct"
            if code in DICOM_CODES:
                return DICOM_CODES[code]
        except Exception:
            pass
    name = path.name.lower()
    for key in sorted(MODALITIES, key=len, reverse=True):
        if key in name:
            return key
    raise ValueError(f"cannot infer modality for {path.name}; pass --modality")
