"""Unit test untuk logika taksonomi & compliance (tidak butuh model .pt)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import (
    PPE_CLASSES,
    SELECTABLE_CATEGORIES,
    Detection,
    DetectionResult,
    PPEDetector,
    classify_label,
)


class _FilterOnly(PPEDetector):
    """Akses `_finalize` tanpa memuat weights."""

    def __init__(self, enabled=None, conf=0.35, category_conf=None):
        self.enabled_categories = enabled
        self.conf = conf
        self.category_conf = category_conf


def make(label: str) -> Detection:
    category, is_violation = classify_label(label)
    return Detection(
        label=label,
        category=category,
        confidence=0.9,
        bbox=[0, 0, 10, 10],
        is_violation=is_violation,
    )


def test_classify_positive_labels():
    assert classify_label("head_helmet") == ("helmet", False)
    assert classify_label("vest") == ("vest", False)
    assert classify_label("boots") == ("shoes", False)
    assert classify_label("Ear-protection") == ("ear_protection", False)


def test_classify_violation_labels():
    assert classify_label("head_nohelmet") == ("helmet", True)
    assert classify_label("face_nomask") == ("mask", True)
    assert classify_label("hand_noglove") == ("glove", True)
    assert classify_label("No_Glasses") == ("glasses", True)
    assert classify_label("Barefoots") == ("shoes", True)
    assert classify_label("Sandals") == ("shoes", True)
    assert classify_label("No_Ear-Protection") == ("ear_protection", True)


def test_classify_unknown_label_falls_back_to_prefix():
    # Model hasil retraining dengan kelas baru tetap tertangani.
    assert classify_label("no_faceshield") == ("faceshield", True)
    assert classify_label("faceshield") == ("faceshield", False)


def test_person_is_not_a_ppe_category():
    assert classify_label("person") == ("person", False)
    assert "person" not in PPE_CLASSES


def test_compliance_empty_frame():
    status = PPEDetector._compute_compliance([])
    assert set(status) == set(PPE_CLASSES)
    assert all(v == "TIDAK TERDETEKSI" for v in status.values())


def test_compliance_detected_and_violation():
    status = PPEDetector._compute_compliance(
        [make("head_helmet"), make("vest"), make("hand_noglove")]
    )
    assert status["helmet"] == "TERDETEKSI"
    assert status["vest"] == "TERDETEKSI"
    assert status["glove"] == "PELANGGARAN"
    assert status["mask"] == "TIDAK TERDETEKSI"


def test_violation_wins_over_positive():
    # Satu orang pakai helm, satu lagi tidak -> frame ditandai PELANGGARAN.
    status = PPEDetector._compute_compliance([make("head_helmet"), make("head_nohelmet")])
    assert status["helmet"] == "PELANGGARAN"


def _result(*labels: str) -> DetectionResult:
    return DetectionResult(detections=[make(l) for l in labels], width=100, height=100)


def test_filter_none_keeps_everything():
    out = _FilterOnly(None)._finalize(_result("head_nohelmet", "hand_noglove"))
    assert {d.label for d in out.detections} == {"head_nohelmet", "hand_noglove"}
    assert set(out.compliance) == set(PPE_CLASSES)


def test_filter_drops_disabled_category():
    # User uncheck "sarung tangan" -> deteksi glove hilang total.
    enabled = set(PPE_CLASSES) - {"glove"}
    out = _FilterOnly(enabled)._finalize(_result("head_nohelmet", "hand_noglove"))

    assert {d.label for d in out.detections} == {"head_nohelmet"}
    assert "glove" not in out.compliance
    assert out.compliance["helmet"] == "PELANGGARAN"


def test_filter_empty_selection_yields_nothing():
    out = _FilterOnly(set())._finalize(_result("head_helmet", "vest"))
    assert out.detections == []
    assert out.compliance == {}


def test_filter_can_hide_person_boxes():
    enabled = set(PPE_CLASSES)  # tanpa "person"
    out = _FilterOnly(enabled)._finalize(_result("person", "vest"))
    assert {d.label for d in out.detections} == {"vest"}


def test_person_is_selectable_but_not_a_ppe_class():
    assert "person" in SELECTABLE_CATEGORIES
    assert "person" not in PPE_CLASSES


def _det(label: str, confidence: float) -> Detection:
    category, is_violation = classify_label(label)
    return Detection(
        label=label,
        category=category,
        confidence=confidence,
        bbox=[0, 0, 10, 10],
        is_violation=is_violation,
    )


def test_detection_floor_without_overrides_is_global_conf():
    assert _FilterOnly(conf=0.4).detection_floor == 0.4


def test_detection_floor_follows_lowest_category_threshold():
    # Model harus dijalankan di ambang terendah, kalau tidak deteksi glove
    # sudah hilang sebelum sempat difilter.
    det = _FilterOnly(conf=0.5, category_conf={"glove": 0.2})
    assert det.detection_floor == 0.2
    assert det.threshold_for("glove") == 0.2
    assert det.threshold_for("helmet") == 0.5


def test_category_conf_lowers_threshold_for_one_category_only():
    det = _FilterOnly(conf=0.5, category_conf={"glove": 0.2})
    out = det._finalize(
        DetectionResult(detections=[_det("hand_noglove", 0.30), _det("head_nohelmet", 0.35)])
    )
    assert {d.label for d in out.detections} == {"hand_noglove"}
    assert out.compliance["glove"] == "PELANGGARAN"
    assert out.compliance["helmet"] == "TIDAK TERDETEKSI"


def test_category_conf_can_also_raise_threshold():
    det = _FilterOnly(conf=0.3, category_conf={"helmet": 0.7})
    out = det._finalize(DetectionResult(detections=[_det("head_helmet", 0.5)]))
    assert out.detections == []


def test_boots_and_barefoot_map_to_same_category():
    assert classify_label("boots")[0] == classify_label("Barefoots")[0] == "shoes"
    status = PPEDetector._compute_compliance([make("boots"), make("Barefoots")])
    assert status["shoes"] == "PELANGGARAN"
