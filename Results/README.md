# Results

`Code/run.py` writes two files per study to this folder:

- `<study>_report.md`: findings (each sentence cites `[F1]`, `[F2]`, ...), impression, and an evidence table
  (ID, label, confidence, location, bounding box). Sentences without a valid citation are marked ⚠️ for review.
- `<study>_evidence.png`: the image with each finding's box and ID drawn, so every citation maps to a visible region.

Example `_report.md` produced by the template backend from two detections (illustrative, not a patient result):

```markdown
## CHEST X-RAY report

### Findings
- Possible nodule in the upper image-left chest (confidence 91%) [F1].
- Possible effusion in the lower image-right chest (confidence 77%) [F2].

### Impression
Findings suggestive of effusion, nodule; clinical correlation recommended.

### Evidence
| ID | Finding | Confidence | Location | BBox |
|---|---|---|---|---|
| F1 | nodule | 0.91 | upper image-left chest | (10, 10, 40, 40) |
| F2 | effusion | 0.77 | lower image-right chest | (60, 70, 120, 120) |
```
