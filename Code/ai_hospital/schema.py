from dataclasses import asdict, dataclass, field
from typing import List, Optional, Tuple


@dataclass
class Finding:
    """One image-grounded observation. `id` (F1, F2, ...) is what report sentences cite."""

    id: str
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    location: str = ""
    area_px: Optional[int] = None  # from the segmentation mask, when available
    source: str = "detector"

    def to_dict(self):
        return asdict(self)


@dataclass
class ReportSentence:
    text: str
    finding_ids: List[str]
    grounded: bool


@dataclass
class Report:
    modality: str
    findings: List[Finding]
    sentences: List[ReportSentence] = field(default_factory=list)
    impression: str = ""

    @property
    def ungrounded(self):
        return [s for s in self.sentences if not s.grounded]

    def to_markdown(self):
        lines = [f"## {self.modality.upper()} report", "", "### Findings"]
        for s in self.sentences:
            flag = "" if s.grounded else " ⚠️ *(not linked to an image region; verify)*"
            lines.append(f"- {s.text}{flag}")
        if self.impression:
            lines += ["", "### Impression", self.impression]
        lines += ["", "### Evidence", "| ID | Finding | Confidence | Location | BBox |", "|---|---|---|---|---|"]
        for f in self.findings:
            lines.append(f"| {f.id} | {f.label} | {f.confidence:.2f} | {f.location} | {tuple(round(v) for v in f.bbox)} |")
        return "\n".join(lines)
