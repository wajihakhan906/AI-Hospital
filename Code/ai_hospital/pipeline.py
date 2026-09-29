from pathlib import Path

from .imaging import load_image
from .modalities import MODALITIES, detect_modality
from .models import UNetSegmenter, YOLODetector
from .report import OllamaLLM, TemplateLLM, generate_report
from .visualize import draw_findings


class DiagnosticPipeline:
    """image -> modality -> detection -> segmentation -> grounded report + evidence overlay."""

    def __init__(self, llm="ollama:llama3", weights_root="."):
        self.weights_root = Path(weights_root)
        self._detectors, self._segmenters = {}, {}
        if llm == "template":
            self.llm = TemplateLLM()
        elif llm.startswith("ollama:"):
            self.llm = OllamaLLM(model=llm.split(":", 1)[1])
        else:
            raise ValueError("llm must be 'template' or 'ollama:<model>'")

    def _detector(self, cfg):
        if cfg.key not in self._detectors:
            self._detectors[cfg.key] = YOLODetector(self.weights_root / cfg.detector_weights)
        return self._detectors[cfg.key]

    def _segmenter(self, cfg):
        if not cfg.segmenter_weights or not (self.weights_root / cfg.segmenter_weights).exists():
            return None
        if cfg.key not in self._segmenters:
            self._segmenters[cfg.key] = UNetSegmenter(self.weights_root / cfg.segmenter_weights)
        return self._segmenters[cfg.key]

    def run(self, path, modality=None, detector=None):
        key = detect_modality(path, modality)
        cfg = MODALITIES[key]
        image = load_image(path)
        findings = (detector or self._detector(cfg))(image, cfg.anatomy_hint)
        seg = self._segmenter(cfg)
        if seg is not None:
            findings = seg(image, findings)
        report = generate_report(cfg.name, findings, self.llm)
        return report, draw_findings(image, findings)
