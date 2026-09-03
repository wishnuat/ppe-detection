"""Unit test absensi wajah: pendaftaran, pencocokan, dan log kehadiran.

Vektor wajah dipalsukan (basis ortonormal + noise terkendali) supaya
perilakunya bisa dikunci tanpa model 37 MB dan tanpa foto orang sungguhan.
Yang diuji di sini adalah kebijakannya — kapan seseorang dinyatakan dikenali,
kapan kehadirannya dicatat, apa yang terjadi saat karyawan dihapus — dan itu
semua tidak bergantung pada dari mana vektornya berasal.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.fatigue.attendance import AttendanceBook
from src.fatigue.types import Identity

DIM = 128


def unit(seed: int) -> np.ndarray:
    """Vektor satuan acak deterministik — berperan sebagai 'wajah orang X'."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


def nudge(vec: np.ndarray, amount: float, seed: int = 0) -> np.ndarray:
    """Versi sedikit berbeda dari wajah yang sama (pose/cahaya lain)."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=DIM).astype(np.float32)
    noise -= noise.dot(vec) * vec          # tegak lurus, supaya `amount` terkontrol
    noise /= np.linalg.norm(noise)
    out = np.sqrt(1 - amount ** 2) * vec + amount * noise
    return out / np.linalg.norm(out)


@pytest.fixture()
def book(tmp_path: Path) -> AttendanceBook:
    return AttendanceBook(db_path=tmp_path / "att.db", threshold=0.40, reentry_gap=300.0)


# ---------- pendaftaran ----------
def test_empty_book_recognizes_nobody(book: AttendanceBook):
    identity = book.identify(unit(1))
    assert not identity.is_known
    assert identity.employee_id is None


def test_enrolled_face_is_recognized(book: AttendanceBook):
    face = unit(1)
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", face)

    identity = book.identify(face)
    assert identity.is_known
    assert identity.employee_id == "EMP001"
    assert identity.name == "Budi"
    assert identity.similarity == pytest.approx(1.0, abs=1e-4)


def test_stranger_is_not_recognized(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", unit(1))
    # Vektor acak lain di 128 dimensi hampir pasti nyaris ortogonal.
    assert not book.identify(unit(999)).is_known


def test_same_person_different_pose_still_recognized(book: AttendanceBook):
    face = unit(1)
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", face)
    # Kemiripan ~0.8: pose/cahaya berbeda, orang yang sama.
    assert book.identify(nudge(face, 0.6)).is_known


def test_multiple_enrollment_photos_improve_coverage(book: AttendanceBook):
    """Skor = similarity tertinggi antar foto, bukan rata-ratanya.

    Kalau dirata-rata, mendaftarkan pose tambahan justru bisa MENURUNKAN skor
    dan menghukum orang yang mendaftar paling lengkap.
    """
    front = unit(1)
    # amount 0.95 -> cosine ~0.31, di bawah ambang 0.40: pose yang begitu
    # berbeda sampai tidak lagi tertangkap oleh foto depan saja.
    side = nudge(front, 0.95, seed=7)
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", front)
    assert not book.identify(side).is_known        # belum terdaftar dari sisi ini

    book.add_embedding("EMP001", side)
    assert book.identify(side).is_known            # sekarang tertangkap


def test_embedding_requires_existing_employee(book: AttendanceBook):
    with pytest.raises(KeyError):
        book.add_embedding("TIDAK_ADA", unit(1))


def test_zero_vector_is_rejected(book: AttendanceBook):
    """Vektor nol = wajah gagal terdeteksi. Menyimpannya merusak pencocokan."""
    book.add_employee("EMP001", "Budi")
    with pytest.raises(ValueError):
        book.add_embedding("EMP001", np.zeros(DIM, dtype=np.float32))


def test_dimension_mismatch_is_reported_clearly(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", unit(1))
    with pytest.raises(ValueError, match="Dimensi"):
        book.identify(np.ones(512, dtype=np.float32) / np.sqrt(512))


# ---------- siklus hidup karyawan ----------
def test_inactive_employee_is_not_matched(book: AttendanceBook):
    face = unit(1)
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", face)
    book.set_active("EMP001", False)
    assert not book.identify(face).is_known

    book.set_active("EMP001", True)
    assert book.identify(face).is_known


def test_delete_removes_embeddings_too(book: AttendanceBook):
    """Embedding yatim tidak boleh tertinggal — itu data biometrik."""
    face = unit(1)
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", face)
    assert book.delete_employee("EMP001")

    assert not book.identify(face).is_known
    assert book.stats()["embeddings"] == 0


def test_re_adding_employee_updates_name(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", unit(1))
    book.add_employee("EMP001", "Budi Santoso", "Produksi")

    identity = book.identify(unit(1))
    assert identity.name == "Budi Santoso"
    assert book.list_employees()[0].num_embeddings == 1   # tidak terduplikasi


def test_list_employees_counts_embeddings(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_employee("EMP002", "Ani")
    for i in range(3):
        book.add_embedding("EMP001", unit(100 + i))

    by_id = {e.employee_id: e for e in book.list_employees()}
    assert by_id["EMP001"].num_embeddings == 3
    assert by_id["EMP002"].num_embeddings == 0


# ---------- log kehadiran ----------
def test_check_in_records_known_person(book: AttendanceBook):
    identity = Identity("EMP001", "Budi", 0.9, True)
    book.add_employee("EMP001", "Budi")
    record = book.check_in(identity, camera="gerbang", now=1000.0)

    assert record is not None
    assert record.employee_id == "EMP001"
    assert record.camera == "gerbang"
    assert len(book.records()) == 1


def test_check_in_ignores_unknown_person(book: AttendanceBook):
    assert book.check_in(Identity.unknown(0.2), now=1000.0) is None
    assert book.records() == []


def test_continuous_presence_records_one_arrival(book: AttendanceBook):
    """Orang yang berdiri di depan kamera = satu baris, berapa pun lamanya."""
    identity = Identity("EMP001", "Budi", 0.9, True)
    book.add_employee("EMP001", "Budi")

    assert book.check_in(identity, now=1000.0) is not None
    for t in range(1, 250):
        assert book.check_in(identity, now=1000.0 + t * 0.04) is None
    assert len(book.records()) == 1


def test_returning_after_absence_is_a_new_arrival(book: AttendanceBook):
    """Pergi makan siang lalu kembali = kedatangan kedua."""
    identity = Identity("EMP001", "Budi", 0.9, True)
    book.add_employee("EMP001", "Budi")
    book.check_in(identity, now=1000.0)
    assert book.check_in(identity, now=1000.0 + 301.0) is not None
    assert len(book.records()) == 2


def test_long_shift_without_absence_stays_one_row(book: AttendanceBook):
    """Dua jam hadir terus-menerus tetap satu baris.

    Cooldown sederhana akan menghasilkan satu baris tiap 5 menit — 24 baris
    untuk satu shift 2 jam. Log absensi yang isinya begitu tidak menjawab
    pertanyaan apa pun; yang dicari orang adalah jam kedatangan.
    """
    identity = Identity("EMP001", "Budi", 0.9, True)
    book.add_employee("EMP001", "Budi")
    for i in range(2 * 3600):          # terlihat tiap detik selama 2 jam
        book.check_in(identity, now=1000.0 + i)
    assert len(book.records()) == 1


def test_cooldown_is_measured_from_last_sighting_not_last_row(book: AttendanceBook):
    """Yang menentukan adalah kapan terakhir TERLIHAT, bukan terakhir dicatat."""
    identity = Identity("EMP001", "Budi", 0.9, True)
    book.add_employee("EMP001", "Budi")
    book.check_in(identity, now=1000.0)
    # Terlihat terus tiap 100 detik selama 10 menit — tidak pernah absen
    # selama 300 detik, jadi tidak ada kedatangan baru meski total > 300 dtk.
    for i in range(1, 7):
        book.check_in(identity, now=1000.0 + i * 100)
    assert len(book.records()) == 1


def test_cooldown_is_per_person(book: AttendanceBook):
    """Dua karyawan yang lewat bersamaan harus dua-duanya tercatat."""
    book.add_employee("EMP001", "Budi")
    book.add_employee("EMP002", "Ani")
    assert book.check_in(Identity("EMP001", "Budi", 0.9, True), now=1000.0) is not None
    assert book.check_in(Identity("EMP002", "Ani", 0.9, True), now=1000.5) is not None
    assert len(book.records()) == 2


def test_records_can_be_filtered(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_employee("EMP002", "Ani")
    book.check_in(Identity("EMP001", "Budi", 0.9, True), now=1000.0)
    book.check_in(Identity("EMP002", "Ani", 0.9, True), now=2000.0)

    assert len(book.records(employee_id="EMP001")) == 1
    assert len(book.records(since=1500.0)) == 1


def test_stats_reflect_database_contents(book: AttendanceBook):
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", unit(1))
    book.add_employee("EMP002", "Ani")
    book.set_active("EMP002", False)

    stats = book.stats()
    assert stats["employees"] == 2
    assert stats["active_employees"] == 1
    assert stats["embeddings"] == 1


def test_data_survives_reopening(tmp_path: Path):
    """Database harus persisten — restart layanan tidak boleh menghapus absensi."""
    path = tmp_path / "att.db"
    face = unit(1)

    first = AttendanceBook(db_path=path)
    first.add_employee("EMP001", "Budi")
    first.add_embedding("EMP001", face)
    first.check_in(Identity("EMP001", "Budi", 0.9, True), now=1000.0)

    second = AttendanceBook(db_path=path)
    assert second.identify(face).is_known
    assert len(second.records()) == 1
