"""AI Hospital: modality-agnostic detection -> segmentation -> grounded report generation."""
from .pipeline import DiagnosticPipeline
from .schema import Finding, Report

__all__ = ["DiagnosticPipeline", "Finding", "Report"]
