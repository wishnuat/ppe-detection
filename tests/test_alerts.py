"""Unit test kebijakan alert & statistik sesi (tidak butuh model .pt)."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertEngine, SessionStats, events_to_csv
from src.detector import Detection, DetectionResult, PPEDetector, classify_label


def make(label: str, confidence: float = 0.9) -> Detection:
    category, is_violation = classify_label(label)
    return Detection(
        label=label,
        category=category,
        confidence=confidence,
        bbox=[0, 0, 10, 10],
        is_violation=is_violation,
    )


def result(*labels: str) -> DetectionResult:
    dets = [make(l) for l in labels]
    return DetectionResult(
        detections=dets,
        compliance=PPEDetector._compute_compliance(dets),
        width=100,
        height=100,
    )


# ---------- debounce ----------
def test_alert_needs_consecutive_frames():
    engine = AlertEngine(min_frames=3, cooldown=0.0)
    assert engine.update(result("head_nohelmet"), now=0.0) == []
    assert engine.update(result("head_nohelmet"), now=0.1) == []
    fired = engine.update(result("head_nohelmet"), now=0.2)
    assert [e.category for e in fired] == ["helmet"]


def test_streak_resets_when_violation_disappears():
    engine = AlertEngine(min_frames=3, cooldown=0.0)
    engine.update(result("head_nohelmet"), now=0.0)
    engine.update(result("head_nohelmet"), now=0.1)
    engine.update(result("head_helmet"), now=0.2)  # sekejap patuh -> streak hangus
    assert engine.update(result("head_nohelmet"), now=0.3) == []
    assert engine.active == set()


# ---------- cooldown ----------
def test_cooldown_suppresses_repeat_alert():
    engine = AlertEngine(min_frames=1, cooldown=10.0)
    assert len(engine.update(result("face_nomask"), now=0.0)) == 1
    assert engine.update(result("face_nomask"), now=5.0) == []
    assert len(engine.update(result("face_nomask"), now=10.0)) == 1
    assert len(engine.events) == 2


def test_active_stays_true_during_cooldown():
    """Cooldown membungkam alarm baru, tapi banner status harus tetap merah."""
    engine = AlertEngine(min_frames=1, cooldown=10.0)
    engine.update(result("face_nomask"), now=0.0)
    engine.update(result("face_nomask"), now=1.0)
    assert engine.active == {"mask"}


# ---------- pemilihan kategori ----------
def test_only_selected_categories_trigger_alert():
    engine = AlertEngine(categories={"helmet"}, min_frames=1, cooldown=0.0)
    fired = engine.update(result("head_nohelmet", "hand_noglove"))
    assert [e.category for e in fired] == ["helmet"]


def test_muting_all_categories_yields_no_alert():
    engine = AlertEngine(categories=set(), min_frames=1, cooldown=0.0)
    assert engine.update(result("head_nohelmet")) == []


def test_configure_changes_policy_without_losing_history():
    engine = AlertEngine(min_frames=1, cooldown=0.0)
    engine.update(result("head_nohelmet"))
    engine.configure(categories={"mask"}, min_frames=2)
    assert len(engine.events) == 1
    assert not engine.watches("helmet")


# ---------- gating orang ----------
def test_require_person_suppresses_alert_without_person():
    engine = AlertEngine(min_frames=1, cooldown=0.0, require_person=True)
    assert engine.update(result("head_nohelmet")) == []
    fired = engine.update(result("head_nohelmet", "person"))
    assert [e.category for e in fired] == ["helmet"]


# ---------- isi event ----------
def test_event_uses_highest_violation_confidence():
    dets = [make("head_nohelmet", 0.4), make("head_nohelmet", 0.82)]
    res = DetectionResult(
        detections=dets, compliance=PPEDetector._compute_compliance(dets)
    )
    engine = AlertEngine(min_frames=1, cooldown=0.0)
    assert engine.update(res)[0].confidence == 0.82


def test_csv_export_has_header_and_rows():
    engine = AlertEngine(min_frames=1, cooldown=0.0)
    engine.update(result("head_nohelmet"), now=0.0)
    engine.update(result("hand_noglove"), now=1.0)
    lines = events_to_csv(engine.events).strip().splitlines()
    assert lines[0].startswith("waktu,kategori,confidence")
    assert len(lines) == 3


def test_reset_clears_everything():
    engine = AlertEngine(min_frames=1, cooldown=0.0)
    engine.update(result("head_nohelmet"))
    engine.reset()
    assert engine.events == [] and engine.active == set() and engine.last_fired == {}


# ---------- statistik sesi ----------
def test_session_stats_compliance_rate():
    stats = SessionStats()
    stats.update(result("head_helmet"), now=0.0)
    stats.update(result("head_nohelmet"), now=1.0)
    stats.update(result("head_helmet"), now=2.0)
    stats.update(result("head_helmet"), now=3.0)
    assert stats.frames == 4
    assert stats.violation_frames == 1
    assert stats.compliance_rate == 75.0
    assert stats.duration == 3.0
    assert stats.per_category_frames["helmet"] == 1


def test_session_stats_empty_is_fully_compliant():
    assert SessionStats().compliance_rate == 100.0
