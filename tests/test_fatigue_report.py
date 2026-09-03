"""Test pencatatan riwayat fatigue dan penyusunan laporan.

Data dibuat sintetis dan waktunya disuntik eksplisit, jadi seluruh test ini
deterministik dan selesai dalam hitungan milidetik — tidak perlu kamera, model,
maupun menunggu satu shift berlalu.

Yang paling penting diuji di sini adalah **arti angkanya**, bukan sekadar
apakah file-nya jadi. Laporan yang formatnya rapi tapi rata-ratanya salah jauh
lebih berbahaya daripada laporan yang gagal dibuat: yang satu ketahuan, yang
lain dipakai untuk mengambil keputusan.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from src.fatigue.attendance import AttendanceBook
from src.fatigue.records import FatigueLog
from src.fatigue.report import _fmt_duration, build_report, to_excel
from src.fatigue.types import FatigueLevel, Identity, PersonState

INTERVAL = 30.0


def person(name: str = "Budi", employee_id: str | None = "EMP001",
           level: FatigueLevel = FatigueLevel.ALERT, perclos: float = 0.04,
           **kwargs) -> PersonState:
    return PersonState(
        identity=Identity(employee_id, name, 0.9, employee_id is not None),
        level=level, perclos=perclos, **kwargs,
    )


def midnight(offset_days: int = 0) -> float:
    now = time.localtime()
    base = time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))
    return base + offset_days * 86400


@pytest.fixture()
def log(tmp_path: Path) -> FatigueLog:
    return FatigueLog(db_path=tmp_path / "att.db", sample_interval=INTERVAL,
                      camera="kamera-1")


# ---------- pencuplikan ----------
def test_samples_are_written_at_the_configured_interval(log: FatigueLog):
    """Dipanggil tiap frame, ditulis tiap interval — bukan tiap frame.

    Satu shift 8 jam pada 9 fps = 260.000 baris per orang kalau tiap frame
    ditulis, dan tidak satu pun pertanyaan laporan butuh resolusi sehalus itu.
    """
    t = midnight() + 3600
    for i in range(600):                 # 600 frame @ 0,5 dtk = 5 menit
        log.record([person()], now=t + i * 0.5)
    # 5 menit / 30 detik = 10 cuplikan (+1 untuk yang pertama).
    assert 10 <= len(log.samples()) <= 11


def test_unknown_level_is_not_sampled(log: FatigueLog):
    """Orang yang belum bisa dinilai tidak boleh mengencerkan rata-rata.

    Menit-menit kalibrasi awal berlevel TIDAK_DIKETAHUI; memasukkannya ke
    rata-rata PERCLOS akan menurunkan angka orang yang benar-benar terukur.
    """
    t = midnight() + 3600
    for i in range(20):
        log.record([person(level=FatigueLevel.UNKNOWN)], now=t + i * INTERVAL)
    assert log.samples() == []


def test_samples_are_per_person(log: FatigueLog):
    t = midnight() + 3600
    for i in range(4):
        log.record([person("Budi", "EMP001"), person("Ani", "EMP002")],
                   now=t + i * INTERVAL)
    names = {s["display_name"] for s in log.samples()}
    assert names == {"Budi", "Ani"}


# ---------- kejadian ----------
def test_only_escalations_are_logged_as_events(log: FatigueLog):
    """Turun level tidak menarik siapa pun dan cuma menggandakan log."""
    t = midnight() + 3600
    log.record([person(level=FatigueLevel.ALERT)], now=t)
    log.record([person(level=FatigueLevel.SEVERE)], now=t + 1)   # naik  -> dicatat
    log.record([person(level=FatigueLevel.ALERT)], now=t + 2)    # turun -> tidak
    log.record([person(level=FatigueLevel.MILD)], now=t + 3)     # naik  -> dicatat

    events = log.events()
    assert [e.level for e in events] == ["WASPADA", "LELAH"] or \
           [e.level for e in events] == ["LELAH", "WASPADA"]
    assert len(events) == 2


def test_staying_at_the_same_level_logs_once(log: FatigueLog):
    t = midnight() + 3600
    for i in range(50):
        log.record([person(level=FatigueLevel.SEVERE)], now=t + i)
    assert len(log.events()) == 1


def test_alert_level_alone_is_not_an_event(log: FatigueLog):
    """SEGAR bukan kabar buruk; ia tidak perlu masuk log kejadian."""
    t = midnight() + 3600
    for i in range(10):
        log.record([person(level=FatigueLevel.ALERT)], now=t + i)
    assert log.events() == []


def test_event_carries_the_reasons(log: FatigueLog):
    """Alert tanpa alasan akan diabaikan — alasannya harus ikut tersimpan."""
    t = midnight() + 3600
    log.record([person(level=FatigueLevel.CRITICAL,
                       reasons=["mata terpejam 4.0 dtk", "PERCLOS 60%"])], now=t)
    event = log.events()[0]
    assert "terpejam" in event.reasons and "PERCLOS" in event.reasons
    assert event.camera == "kamera-1"


def test_reset_session_allows_a_fresh_escalation(log: FatigueLog):
    """Sesi baru harus bisa melaporkan lagi meski levelnya sama seperti kemarin."""
    t = midnight() + 3600
    log.record([person(level=FatigueLevel.SEVERE)], now=t)
    log.reset_session()
    log.record([person(level=FatigueLevel.SEVERE)], now=t + 10)
    assert len(log.events()) == 2


# ---------- agregasi laporan ----------
def _seed_day(db: Path, minutes_alert: int, minutes_severe: int,
              perclos_alert: float = 0.04, perclos_severe: float = 0.40) -> None:
    log = FatigueLog(db_path=db, sample_interval=INTERVAL)
    t = midnight() + 8 * 3600
    for i in range(minutes_alert * 2):          # 2 cuplikan per menit
        log.record([person(level=FatigueLevel.ALERT, perclos=perclos_alert)],
                   now=t + i * INTERVAL)
    t += minutes_alert * 60
    for i in range(minutes_severe * 2):
        log.record([person(level=FatigueLevel.SEVERE, perclos=perclos_severe)],
                   now=t + i * INTERVAL)


def test_observed_time_counts_samples_not_wall_clock(tmp_path: Path):
    """Waktu saat orangnya tidak terlihat tidak boleh dihitung terpantau.

    Kalau lama pengamatan dihitung dari selisih jam pertama dan terakhir,
    seseorang yang datang jam 8 lalu pulang dan kembali jam 4 akan tercatat
    "terpantau 8 jam" — padahal sistem tidak melihat apa pun di antaranya.
    """
    db = tmp_path / "att.db"
    log = FatigueLog(db_path=db, sample_interval=INTERVAL)
    t = midnight() + 8 * 3600
    for i in range(10):                          # 10 cuplikan = 5 menit
        log.record([person()], now=t + i * INTERVAL)
    log.record([person()], now=t + 6 * 3600)     # muncul lagi 6 jam kemudian

    report = build_report(db_path=db, sample_interval=INTERVAL)
    day = report.person_days[0]
    assert day.observed_seconds == pytest.approx(11 * INTERVAL)
    assert day.observed_seconds < 6 * 3600       # bukan selisih jam


def test_time_at_each_level_adds_up(tmp_path: Path):
    db = tmp_path / "att.db"
    _seed_day(db, minutes_alert=30, minutes_severe=10)

    report = build_report(db_path=db, sample_interval=INTERVAL)
    day = report.person_days[0]
    assert day.seconds_at["SEGAR"] == pytest.approx(30 * 60)
    assert day.seconds_at["LELAH"] == pytest.approx(10 * 60)
    assert day.observed_seconds == pytest.approx(40 * 60)
    assert day.worst_level is FatigueLevel.SEVERE


def test_perclos_mean_reflects_the_whole_day(tmp_path: Path):
    db = tmp_path / "att.db"
    _seed_day(db, minutes_alert=30, minutes_severe=10,
              perclos_alert=0.05, perclos_severe=0.45)

    report = build_report(db_path=db, sample_interval=INTERVAL)
    day = report.person_days[0]
    expected = (60 * 0.05 + 20 * 0.45) / 80      # 60 vs 20 cuplikan
    assert day.perclos_mean == pytest.approx(expected, abs=1e-6)
    assert day.perclos_max == pytest.approx(0.45)


def test_team_summary_weights_perclos_by_time_observed(tmp_path: Path):
    """Hari yang cuma terpantau 5 menit tidak boleh setara dengan hari 8 jam."""
    db = tmp_path / "att.db"
    log = FatigueLog(db_path=db, sample_interval=INTERVAL)

    # Hari ini: 100 cuplikan PERCLOS 0.02
    t = midnight() + 8 * 3600
    for i in range(100):
        log.record([person(perclos=0.02)], now=t + i * INTERVAL)
    # Kemarin: 2 cuplikan PERCLOS 0.50
    t = midnight(-1) + 8 * 3600
    log.reset_session()
    for i in range(2):
        log.record([person(perclos=0.50)], now=t + i * INTERVAL)

    report = build_report(db_path=db, start=midnight(-1), end=midnight(1),
                          sample_interval=INTERVAL)
    row = report.team_summary()[0]
    expected = (100 * 0.02 + 2 * 0.50) / 102
    assert row["PERCLOS rata2"] == pytest.approx(round(expected, 4), abs=1e-4)
    # Rata-rata dari rata-rata harian akan memberi (0.02 + 0.50) / 2 = 0.26.
    assert row["PERCLOS rata2"] < 0.10


def test_report_outside_range_is_empty(tmp_path: Path):
    db = tmp_path / "att.db"
    _seed_day(db, minutes_alert=10, minutes_severe=0)

    report = build_report(db_path=db, start=midnight(-5), end=midnight(-4))
    assert report.person_days == []
    assert any("Tidak ada cuplikan" in n for n in report.notes)


def test_unknown_person_is_reported_separately(tmp_path: Path):
    db = tmp_path / "att.db"
    log = FatigueLog(db_path=db, sample_interval=INTERVAL)
    t = midnight() + 3600
    for i in range(4):
        log.record([person("Tidak dikenal", employee_id=None)], now=t + i * INTERVAL)

    report = build_report(db_path=db, sample_interval=INTERVAL)
    assert report.person_days[0].employee_id == "(tidak dikenal)"
    assert any("tanpa identitas" in n for n in report.notes)


# ---------- Excel ----------
def test_excel_has_every_expected_sheet(tmp_path: Path):
    openpyxl = pytest.importorskip("openpyxl")
    db = tmp_path / "att.db"
    _seed_day(db, minutes_alert=20, minutes_severe=5)
    AttendanceBook(db_path=db).add_employee("EMP001", "Budi", "Produksi")

    report = build_report(db_path=db, sample_interval=INTERVAL)
    path = to_excel(report, tmp_path / "laporan.xlsx")

    assert path.exists() and path.stat().st_size > 0
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == [
        "Ringkasan", "Rekap per orang", "Harian per orang",
        "Log absensi", "Kejadian fatigue", "Karyawan",
    ]
    ws = wb["Rekap per orang"]
    assert ws["A1"].value == "ID"          # baris pertama = header
    assert ws.freeze_panes == "A2"         # header tetap terlihat saat di-scroll


def test_excel_is_written_even_with_no_data(tmp_path: Path):
    """Laporan kosong harus tetap jadi file, dengan keterangan kenapa kosong.

    Melempar error untuk periode yang memang belum ada datanya akan membuat
    orang mengira sistemnya rusak.
    """
    pytest.importorskip("openpyxl")
    report = build_report(db_path=tmp_path / "att.db",
                          start=midnight(-9), end=midnight(-8))
    path = to_excel(report, tmp_path / "kosong.xlsx")
    assert path.exists()


# ---------- format ----------
@pytest.mark.parametrize("seconds,expected", [
    (0, "0d"), (45, "45d"), (90, "1m"), (600, "10m"),
    (3600, "1j 0m"), (7500, "2j 5m"),
])
def test_duration_formatting(seconds, expected):
    assert _fmt_duration(seconds) == expected


def test_negative_duration_does_not_crash():
    assert _fmt_duration(-5) == "0d"
