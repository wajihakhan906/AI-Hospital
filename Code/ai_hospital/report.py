"""Grounded report generation.

The LLM only sees the structured findings table and must cite a finding ID ([F1], [F2], ...)
in every sentence. After generation each sentence is checked: citations must refer to real
findings, and sentences without a valid citation are flagged as ungrounded for the clinician.
"""
import json
import re
import urllib.request

from .schema import Report, ReportSentence

CITATION = re.compile(r"\[(F\d+)\]")

SYSTEM_PROMPT = """You are a radiology reporting assistant. Write the FINDINGS section of a {modality} report
using ONLY the detections listed below. Every sentence MUST end with the ID(s) of the detection(s) it
describes in square brackets, e.g. "There is a 2 cm opacity in the right lower zone [F1]."
Do not mention anything that is not in the list. If the list is empty, write
"No abnormality detected by the automated system." Then write one line starting with "IMPRESSION:".

Detections:
{table}
"""


def findings_table(findings):
    if not findings:
        return "(none)"
    return "\n".join(f"{f.id}: {f.label}, confidence {f.confidence:.2f}, location {f.location}"
                     + (f", segmented area {f.area_px} px" if f.area_px else "") for f in findings)


class OllamaLLM:
    """Local LLM through the Ollama HTTP API (e.g. `ollama pull llama3`)."""

    def __init__(self, model="llama3", host="http://localhost:11434"):
        self.model, self.host = model, host

    def __call__(self, prompt):
        body = json.dumps({"model": self.model, "prompt": prompt, "stream": False,
                           "options": {"temperature": 0.1}}).encode()
        req = urllib.request.Request(f"{self.host}/api/generate", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read())["response"]


class TemplateLLM:
    """Deterministic fallback without an LLM: one templated sentence per finding."""

    def __call__(self, prompt):
        rows = re.findall(r"^(F\d+): (.+?), confidence ([\d.]+), location (.+?)(?:, segmented|$)", prompt, re.M)
        if not rows:
            return "No abnormality detected by the automated system.\nIMPRESSION: No acute findings."
        lines = [f"Possible {label} in the {loc} (confidence {float(c):.0%}) [{fid}]." for fid, label, c, loc in rows]
        labels = sorted({r[1] for r in rows})
        lines.append(f"IMPRESSION: Findings suggestive of {', '.join(labels)}; clinical correlation recommended.")
        return "\n".join(lines)


def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]


def ground(text, findings):
    valid = {f.id for f in findings}
    impression, sentences = "", []
    for s in split_sentences(text):
        if s.upper().startswith("IMPRESSION:"):
            impression = s.split(":", 1)[1].strip()
            continue
        ids = CITATION.findall(s)
        ok = bool(ids) and all(i in valid for i in ids)
        if not findings and not ids and "no abnormality" in s.lower():
            ok = True
        sentences.append(ReportSentence(text=s, finding_ids=[i for i in ids if i in valid], grounded=ok))
    return sentences, impression


def generate_report(modality_name, findings, llm):
    prompt = SYSTEM_PROMPT.format(modality=modality_name, table=findings_table(findings))
    sentences, impression = ground(llm(prompt), findings)
    return Report(modality=modality_name, findings=findings, sentences=sentences, impression=impression)
