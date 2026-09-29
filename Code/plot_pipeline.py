"""Draws the AI Hospital pipeline to ../Figures/pipeline.png."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

STEPS = [
    ("Input", "DICOM / PNG\n8 modalities", "#8172B2"),
    ("Modality router", "DICOM (0008,0060)\n→ config", "#64B5CD"),
    ("Detection", "YOLO per modality\n→ findings F1…Fn", "#4C72B0"),
    ("Segmentation", "U-Net inside each box\n→ extent / area", "#55A868"),
    ("Grounded report", "LLM cites [Fi] in\nevery sentence", "#C44E52"),
    ("Verification", "uncited / invalid\nsentences flagged", "#DD8452"),
]


def main():
    fig, ax = plt.subplots(figsize=(13, 3.2))
    w, gap = 1.8, 0.35
    for i, (title, body, color) in enumerate(STEPS):
        x = i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, 0), w, 1.4, boxstyle="round,pad=0.05", fc=color, ec="black"))
        ax.text(x + w / 2, 1.05, title, ha="center", va="center", color="white", weight="bold", fontsize=10)
        ax.text(x + w / 2, 0.5, body, ha="center", va="center", color="white", fontsize=8.5)
        if i < len(STEPS) - 1:
            ax.annotate("", (x + w + gap - 0.02, 0.7), (x + w + 0.07, 0.7), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.text(len(STEPS) * (w + gap) / 2 - gap / 2, -0.45,
            "Output: report + evidence overlay; every statement links back to the image region that produced it",
            ha="center", fontsize=9, style="italic")
    ax.set_xlim(-0.2, len(STEPS) * (w + gap)); ax.set_ylim(-0.7, 1.6); ax.axis("off")
    fig.tight_layout(); fig.savefig("../Figures/pipeline.png", dpi=200)


if __name__ == "__main__":
    main()
