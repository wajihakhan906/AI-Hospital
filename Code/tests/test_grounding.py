import numpy as np

from ai_hospital.modalities import detect_modality
from ai_hospital.report import TemplateLLM, generate_report, ground
from ai_hospital.schema import Finding
from ai_hospital.visualize import draw_findings

F = [Finding("F1", "nodule", 0.91, (10, 10, 40, 40), "upper image-left chest"),
     Finding("F2", "effusion", 0.77, (60, 70, 120, 120), "lower image-right chest")]


def test_every_template_sentence_is_grounded():
    r = generate_report("Chest X-ray", F, TemplateLLM())
    assert r.sentences and not r.ungrounded
    assert {i for s in r.sentences for i in s.finding_ids} == {"F1", "F2"}
    assert r.impression


def test_hallucinated_or_uncited_sentences_are_flagged():
    text = "Nodule in the left upper zone [F1]. Cardiomegaly is present. Fracture of rib [F9].\nIMPRESSION: x"
    sents, imp = ground(text, F)
    assert [s.grounded for s in sents] == [True, False, False]
    assert imp == "x"


def test_no_findings_report():
    r = generate_report("CT", [], TemplateLLM())
    assert len(r.sentences) == 1 and r.sentences[0].grounded


def test_modality_from_filename_and_override():
    assert detect_modality("patient_mammography_01.png") == "mammography"
    assert detect_modality("scan.png", "petct") == "petct"


def test_overlay_has_same_size():
    img = np.zeros((128, 128, 3), np.uint8)
    assert draw_findings(img, F).size == (128, 128)


def test_pipeline_end_to_end_with_stub_detector(tmp_path):
    from PIL import Image

    from ai_hospital import DiagnosticPipeline

    path = tmp_path / "chest_xray.png"
    Image.fromarray(np.full((128, 128), 90, np.uint8)).save(path)
    report, overlay = DiagnosticPipeline(llm="template").run(path, detector=lambda img, hint: F)
    assert report.modality == "Chest X-ray" and not report.ungrounded
    assert "| F2 | effusion |" in report.to_markdown()
