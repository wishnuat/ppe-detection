"""Unit test agregasi temporal & fusi fatigue (tidak butuh model sama sekali).

Semua sinyal di sini sintetis: sengaja, karena inti yang diuji adalah
*logikanya* — kapan sesuatu dihitung microsleep, bagaimana PERCLOS dihitung
saat wajah hilang sebentar, kapan level boleh turun. Menguji itu lewat video
asli akan lambat, tidak deterministik, dan justru menyembunyikan kasus tepi
yang paling ingin dikunci.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.fatigue.fusion import FatigueFusion, FusionConfig, FusionWeights
from src.fatigue.temporal import PersonTracker, TemporalSummary
from src.fatigue.types import FatigueLevel, FatigueSignals

FPS = 20.0
DT = 1.0 / FPS


def sig(
    *,
    ear: float | None = 0.30,
    closed: bool = False,
    mouth: bool = False,
    pitch: float = 0.0,
    blink: float | None = None,
) -> FatigueSignals:
    """Sinyal satu frame. Default = orang segar, mata terbuka, menghadap depan."""
    return FatigueSignals(
        ear=ear,
        mar=0.05,
        eye_closed=closed,
        mouth_open=mouth,
        pitch=pitch,
        yaw=0.0,
        roll=0.0,
        blink_score=blink if blink is not None else (0.8 if closed else 0.05),
        jaw_open=0.6 if mouth else 0.02,
    )


def feed(tracker: PersonTracker, signals: list[FatigueSignals],
         cnn: float = 0.0, t0: float = 0.0) -> float:
    """Masukkan urutan sinyal pada laju FPS tetap. Kembalikan waktu terakhir."""
    t = t0
    for s in signals:
        tracker.update(s, cnn, now=t)
        t += DT
    return t - DT


# ---------- PERCLOS ----------
def test_perclos_counts_only_usable_frames():
    """Frame tanpa landmark tidak boleh mengencerkan PERCLOS.

    Kalau ikut dibagi, orang yang sering menoleh selalu terlihat lebih segar
    daripada yang diam menghadap kamera — persis kebalikan dari yang benar.
    """
    tracker = PersonTracker(calibrate=False)
    # 20 frame mata tertutup, 20 frame wajah hilang, 20 frame mata terbuka.
    signals = ([sig(closed=True)] * 20 + [FatigueSignals()] * 20 + [sig()] * 20)
    t = feed(tracker, signals)
    summary = tracker.summarize(now=t)

    # 20 dari 40 frame terpakai -> 50%, bukan 20/60 = 33%.
    assert summary.perclos == pytest.approx(0.5, abs=0.01)
    assert summary.usable_ratio == pytest.approx(40 / 60, abs=0.01)


def test_perclos_zero_when_eyes_open():
    tracker = PersonTracker(calibrate=False)
    t = feed(tracker, [sig()] * 100)
    assert tracker.summarize(now=t).perclos == 0.0


# ---------- microsleep ----------
def test_short_closure_is_blink_not_microsleep():
    tracker = PersonTracker(calibrate=False)
    # 0,3 detik terpejam = kedipan normal.
    signals = [sig()] * 20 + [sig(closed=True)] * 6 + [sig()] * 40
    t = feed(tracker, signals)
    summary = tracker.summarize(now=t)
    assert summary.microsleep_count == 0
    assert summary.blink_rate > 0


def test_long_closure_is_microsleep():
    tracker = PersonTracker(calibrate=False, microsleep_seconds=1.5)
    # 2 detik terpejam.
    signals = [sig()] * 20 + [sig(closed=True)] * 40 + [sig()] * 20
    t = feed(tracker, signals)
    summary = tracker.summarize(now=t)
    assert summary.microsleep_count == 1
    assert summary.longest_closure >= 1.5


def test_ongoing_closure_reported_before_it_ends():
    """Mata yang MASIH terpejam harus dilaporkan sekarang, bukan nanti.

    Kalau durasi baru dicatat saat mata membuka lagi, orang yang benar-benar
    tertidur justru tidak pernah memicu apa pun — kasus paling gawat malah
    jadi kasus yang paling lama diam.
    """
    tracker = PersonTracker(calibrate=False)
    signals = [sig()] * 20 + [sig(closed=True)] * 60   # 3 detik dan belum berhenti
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).longest_closure >= 2.9


def test_face_disappearing_does_not_inflate_closure():
    """Orang yang berbalik badan bukan orang yang tertidur."""
    tracker = PersonTracker(calibrate=False)
    signals = [sig(closed=True)] * 10 + [FatigueSignals()] * 100 + [sig()] * 10
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).longest_closure < 1.0


# ---------- menguap & terkulai ----------
def test_brief_mouth_opening_is_not_a_yawn():
    tracker = PersonTracker(calibrate=False, yawn_seconds=1.5)
    signals = [sig()] * 10 + [sig(mouth=True)] * 10 + [sig()] * 10   # 0,5 dtk
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).yawn_rate == 0.0


def test_sustained_mouth_opening_is_a_yawn():
    tracker = PersonTracker(calibrate=False, yawn_seconds=1.5)
    signals = [sig()] * 10 + [sig(mouth=True)] * 40 + [sig()] * 10   # 2 dtk
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).yawn_rate > 0.0


def test_head_nod_detected_above_pitch_threshold():
    tracker = PersonTracker(calibrate=False, nod_pitch=22.0, nod_seconds=1.2)
    signals = [sig()] * 10 + [sig(pitch=30.0)] * 40 + [sig()] * 10
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).nod_rate > 0.0


def test_looking_slightly_down_is_not_a_nod():
    tracker = PersonTracker(calibrate=False, nod_pitch=22.0)
    signals = [sig(pitch=15.0)] * 100
    t = feed(tracker, signals)
    assert tracker.summarize(now=t).nod_rate == 0.0


# ---------- jendela geser ----------
def test_old_samples_leave_the_window():
    tracker = PersonTracker(window_seconds=5.0, calibrate=False)
    feed(tracker, [sig(closed=True)] * 100, t0=0.0)      # 5 dtk mata tertutup
    t = feed(tracker, [sig()] * 100, t0=5.0)             # 5 dtk mata terbuka
    # Jendela 5 detik hanya menyisakan bagian mata-terbuka.
    assert tracker.summarize(now=t).perclos < 0.05


# ---------- kalibrasi personal ----------
def test_calibration_adapts_threshold_to_the_person():
    """Orang bermata sipit tidak boleh dianggap terpejam terus-menerus."""
    tracker = PersonTracker(calibrate=True)
    # EAR terjaga orang ini cuma 0.20 — di bawah ambang default global 0.21.
    feed(tracker, [sig(ear=0.20, blink=0.05)] * 200)
    assert tracker.calibrated
    assert tracker.ear_threshold is not None
    assert tracker.ear_threshold < 0.20


def test_calibration_ignores_frames_with_eyes_closed():
    """Baseline mata-terbuka tidak boleh tercemar frame mata-tertutup."""
    tracker = PersonTracker(calibrate=True)
    feed(tracker, [sig(ear=0.30, blink=0.05)] * 100 + [sig(ear=0.08, closed=True)] * 100)
    # Baseline ~0.30, jadi ambangnya ~0.216 — jauh di atas EAR terpejam 0.08.
    assert tracker.ear_threshold == pytest.approx(0.30 * 0.72, abs=0.02)


def test_reset_keeps_calibration():
    tracker = PersonTracker(calibrate=True)
    feed(tracker, [sig(ear=0.30, blink=0.05)] * 100)
    threshold = tracker.ear_threshold
    tracker.reset()
    assert tracker.summarize(now=100.0).observed_seconds == 0.0
    assert tracker.ear_threshold == threshold


# ---------- fusi ----------
def summary(**kwargs) -> TemporalSummary:
    base = {"observed_seconds": 60.0, "usable_ratio": 1.0}
    return TemporalSummary(**(base | kwargs))


def test_unreliable_summary_reports_unknown_not_alert():
    """Belum cukup data bukan berarti orangnya segar."""
    fusion = FatigueFusion()
    result = fusion.update(summary(observed_seconds=2.0), now=0.0)
    assert result.level is FatigueLevel.UNKNOWN


def test_mostly_hidden_face_reports_unknown():
    fusion = FatigueFusion()
    result = fusion.update(summary(usable_ratio=0.1), now=0.0)
    assert result.level is FatigueLevel.UNKNOWN


def test_clean_signals_report_alert():
    fusion = FatigueFusion()
    result = fusion.update(summary(cnn_score=0.05, blink_rate=15.0), now=0.0)
    assert result.level is FatigueLevel.ALERT


def test_high_perclos_escalates():
    fusion = FatigueFusion()
    low = fusion.update(summary(perclos=0.05), now=0.0).level
    fusion.reset()
    high = fusion.update(summary(perclos=0.45, cnn_score=0.8), now=0.0).level
    assert high.severity > low.severity


def test_microsleep_overrides_low_soft_score():
    """Satu microsleep sudah cukup, berapa pun rata-rata jendelanya.

    Menunggu skor lunak 60 detik naik untuk kejadian sesaat berarti peringatan
    datang puluhan detik setelah orangnya tertidur.
    """
    fusion = FatigueFusion()
    result = fusion.update(
        summary(perclos=0.02, cnn_score=0.0, microsleep_count=1, longest_closure=1.8),
        now=0.0,
    )
    assert result.level is FatigueLevel.SEVERE


def test_very_long_closure_is_critical():
    fusion = FatigueFusion()
    result = fusion.update(
        summary(microsleep_count=1, longest_closure=4.0), now=0.0
    )
    assert result.level is FatigueLevel.CRITICAL


def test_escalation_is_immediate():
    fusion = FatigueFusion()
    fusion.update(summary(cnn_score=0.0), now=0.0)
    result = fusion.update(summary(perclos=0.5, cnn_score=0.9, yawn_rate=3.0), now=1.0)
    assert result.level.severity >= FatigueLevel.SEVERE.severity


def test_downgrade_waits_for_dwell_time():
    """Level tidak boleh berkedip turun-naik; operator akan berhenti percaya."""
    config = FusionConfig(downgrade_dwell_seconds=20.0)
    fusion = FatigueFusion(config)
    fusion.update(summary(perclos=0.5, cnn_score=0.9, yawn_rate=3.0), now=0.0)
    high = fusion.level

    clean = summary(cnn_score=0.0)
    assert fusion.update(clean, now=5.0).level is high      # belum cukup lama
    assert fusion.update(clean, now=15.0).level is high
    assert fusion.update(clean, now=25.0).level is FatigueLevel.ALERT


def test_downgrade_timer_restarts_if_condition_changes():
    config = FusionConfig(downgrade_dwell_seconds=20.0)
    fusion = FatigueFusion(config)
    fusion.update(summary(perclos=0.5, cnn_score=0.9, yawn_rate=3.0), now=0.0)

    fusion.update(summary(cnn_score=0.0), now=5.0)                  # target ALERT
    middling = summary(perclos=0.30, cnn_score=0.4)                 # target MILD
    fusion.update(middling, now=10.0)                               # target berubah
    # Timer mulai lagi dari t=10, jadi di t=25 (baru 15 dtk) belum boleh turun.
    assert fusion.update(middling, now=25.0).level.severity > FatigueLevel.MILD.severity
    # Setelah 20 detik penuh di target yang sama, baru boleh turun ke MILD.
    assert fusion.update(middling, now=31.0).level is FatigueLevel.MILD


def test_reasons_explain_the_decision():
    fusion = FatigueFusion()
    result = fusion.update(
        summary(perclos=0.35, yawn_rate=2.0, microsleep_count=2, longest_closure=2.0),
        now=0.0,
    )
    joined = " ".join(result.reasons).lower()
    assert "perclos" in joined
    assert "microsleep" in joined
    assert "menguap" in joined


def test_very_low_blink_rate_is_evidence_not_health():
    """Jarang berkedip bukan tanda segar — ia justru tanda kelelahan berat."""
    fusion = FatigueFusion()
    _, contrib_none = fusion.score(summary(blink_rate=0.0))
    _, contrib_low = fusion.score(summary(blink_rate=3.0))
    _, contrib_normal = fusion.score(summary(blink_rate=15.0))
    assert contrib_low["blink"] > contrib_normal["blink"]
    assert contrib_none["blink"] == 0.0        # nol = tidak ada data, bukan sinyal


def test_only_perclos_can_escalate_alone():
    """Invarian inti: tidak ada sumber selain PERCLOS yang boleh berbicara sendiri.

    Classifier menghasilkan distribusi yang hampir biner, jadi kontribusinya
    praktis berupa saklar. Kalau bobotnya mencapai `mild_at`, satu keluaran
    yang keliru-tapi-yakin cukup untuk melaporkan orang yang matanya terbuka
    lebar sepanjang menit itu sebagai waspada — dan tidak ada di dalam sistem
    yang bisa membantahnya.
    """
    config = FusionConfig()
    assert config.sources_that_can_escalate_alone() == ["perclos"]


def test_confident_cnn_alone_does_not_escalate():
    fusion = FatigueFusion()
    result = fusion.update(summary(cnn_score=1.0), now=0.0)
    assert result.level is FatigueLevel.ALERT


def test_high_perclos_alone_does_escalate():
    """PERCLOS dikecualikan dengan sengaja: ia pengukuran, bukan tebakan model."""
    fusion = FatigueFusion()
    result = fusion.update(summary(perclos=0.45), now=0.0)
    assert result.level.severity >= FatigueLevel.MILD.severity


def test_cnn_plus_one_behavioural_signal_does_escalate():
    """Dua sumber yang sepakat memang harus menaikkan level."""
    fusion = FatigueFusion()
    result = fusion.update(summary(cnn_score=1.0, perclos=0.20), now=0.0)
    assert result.level.severity >= FatigueLevel.MILD.severity


def test_ceiling_reports_relaxed_invariant_for_custom_weights():
    """Bobot yang melonggarkan invarian harus bisa dideteksi, bukan diam-diam."""
    config = FusionConfig(
        weights=FusionWeights(cnn=0.55, perclos=0.25, blink=0.05, yawn=0.10, nod=0.05)
    )
    assert "cnn" in config.sources_that_can_escalate_alone()


def test_weights_are_normalized():
    fusion = FatigueFusion(FusionConfig(weights=FusionWeights(cnn=10, perclos=10,
                                                             blink=10, yawn=10, nod=10)))
    score, _ = fusion.score(
        summary(cnn_score=1.0, perclos=1.0, blink_rate=100.0,
                yawn_rate=100.0, nod_rate=100.0)
    )
    assert score == pytest.approx(1.0, abs=1e-6)


def test_score_never_exceeds_one():
    fusion = FatigueFusion()
    score, _ = fusion.score(
        summary(cnn_score=1.0, perclos=5.0, blink_rate=500.0,
                yawn_rate=500.0, nod_rate=500.0)
    )
    assert score <= 1.0


def test_ascending_thresholds_are_enforced():
    with pytest.raises(ValueError):
        FusionConfig(mild_at=0.6, severe_at=0.4, critical_at=0.8)


# ---------- mematikan sumber ----------
def test_without_renormalizes_remaining_weights():
    """Membuang satu sumber tidak boleh menyusutkan skala skor.

    Kalau bobot CNN (0,20) sekadar dibiarkan menyumbang nol, skor maksimum
    yang mungkin jadi 0,80 — sehingga ambang KRITIS 0,70 nyaris mustahil
    tercapai dan seluruh sistem jadi tumpul tanpa ada yang menyadarinya.
    """
    weights = FusionWeights().without("cnn")
    as_dict = weights.as_dict()
    assert as_dict["cnn"] == 0.0
    assert sum(as_dict.values()) == pytest.approx(1.0)


def test_without_keeps_relative_proportions():
    full = FusionWeights().as_dict()
    trimmed = FusionWeights().without("cnn").as_dict()
    # PERCLOS tetap 4x bobot kedip, sebelum maupun sesudah.
    assert trimmed["perclos"] / trimmed["blink"] == pytest.approx(
        full["perclos"] / full["blink"]
    )


def test_invariant_survives_removing_cnn():
    """Hanya PERCLOS yang boleh naik sendirian — juga saat CNN dimatikan."""
    config = FusionConfig(weights=FusionWeights().without("cnn"))
    assert config.sources_that_can_escalate_alone() == ["perclos"]


def test_full_score_reachable_without_cnn():
    fusion = FatigueFusion(FusionConfig(weights=FusionWeights().without("cnn")))
    score, _ = fusion.score(
        summary(perclos=1.0, blink_rate=100.0, yawn_rate=100.0, nod_rate=100.0)
    )
    assert score == pytest.approx(1.0)


def test_removing_unknown_source_is_rejected():
    with pytest.raises(ValueError, match="tidak dikenal"):
        FusionWeights().without("eeg")


def test_removing_every_source_is_rejected():
    with pytest.raises(ValueError):
        FusionWeights().without("cnn", "perclos", "blink", "yawn", "nod")
