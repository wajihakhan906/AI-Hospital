"""Command-line entry point.

    python run.py --image scan.dcm                       # modality from the DICOM header
    python run.py --image chest.png --modality xray --llm template
"""
import argparse
from pathlib import Path

from ai_hospital import DiagnosticPipeline
from ai_hospital.modalities import MODALITIES


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--modality", choices=sorted(MODALITIES))
    ap.add_argument("--llm", default="ollama:llama3", help="'ollama:<model>' or 'template'")
    ap.add_argument("--weights", default=".", help="folder containing weights/*.pt")
    ap.add_argument("--out", default="../Results")
    args = ap.parse_args()

    report, overlay = DiagnosticPipeline(args.llm, args.weights).run(args.image, args.modality)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = Path(args.image).stem
    (out / f"{stem}_report.md").write_text(report.to_markdown())
    overlay.save(out / f"{stem}_evidence.png")
    print(report.to_markdown())
    if report.ungrounded:
        print(f"\n{len(report.ungrounded)} sentence(s) are not linked to image evidence; review before sign-off.")


if __name__ == "__main__":
    main()
