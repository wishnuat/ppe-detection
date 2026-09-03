"""Test validasi & pendaftaran foto wajah — aturan bersama CLI, UI, dan API.

Modul ini ada karena ketiga jalur pendaftaran dulu punya aturannya sendiri dan
ketiganya berbeda. Test di sini mengunci aturan yang berlaku sekarang, dan
salah satunya secara eksplisit memeriksa bahwa ketiga jalur memang memanggil
definisi yang sama — supaya duplikasinya tidak diam-diam tumbuh lagi.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pytest

from src.fatigue.attendance import AttendanceBook
from src.fatigue.enrollment import (
    DUPLICATE_SIM,
    MIN_ENROLL_FACE,
    SOFT_SHARPNESS,
    check_photo,
    enroll_photos,
    face_sharpness,
)
from src.fatigue.types import FaceBox

DIM = 128


def box(x1: int, y1: int, x2: int, y2: int) -> FaceBox:
    return FaceBox(bbox=[x1, y1, x2, y2], confidence=0.9,
                   landmarks=[[0, 0]] * 5, raw=[0.0] * 15)


def noisy(h: int = 300, w: int = 300, seed: int = 0) -> np.ndarray:
    """Gambar berderau — kaya frekuensi tinggi, jadi 'tajam'."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


def flat(h: int = 300, w: int = 300) -> np.ndarray:
    """Gambar rata — nol frekuensi tinggi, jadi 'buram' maksimal."""
    return np.full((h, w, 3), 128, dtype=np.uint8)


class FakeEmbedder:
    backend = "sface"
    dim = DIM

    def __init__(self, vectors: list[np.ndarray] | None = None) -> None:
        self.vectors = vectors or []
        self.calls = 0

    @property
    def threshold(self) -> float:
        return 0.40

    def embed(self, frame, face):
        vec = (self.vectors[self.calls] if self.calls < len(self.vectors)
               else unit(self.calls + 500))
        self.calls += 1
        return vec


class FakeDetector:
    def __init__(self, faces_per_call: list[list[FaceBox]]) -> None:
        self.faces_per_call = faces_per_call
        self.index = 0

    def detect(self, frame):
        if self.index < len(self.faces_per_call):
            faces = self.faces_per_call[self.index]
        else:
            faces = self.faces_per_call[-1] if self.faces_per_call else []
        self.index += 1
        return faces


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


# ---------- penolakan keras ----------
def test_no_face_is_rejected():
    result = check_photo(noisy(), [], name="a.jpg")
    assert not result.accepted
    assert "tidak terdeteksi" in result.reason


def test_multiple_faces_are_rejected():
    """Foto pendaftaran harus satu orang — kalau tidak, embedding siapa?"""
    faces = [box(0, 0, 120, 120), box(150, 0, 270, 120)]
    result = check_photo(noisy(), faces, name="a.jpg")
    assert not result.accepted
    assert "2 wajah" in result.reason


def test_small_face_is_rejected():
    small = MIN_ENROLL_FACE - 10
    result = check_photo(noisy(), [box(0, 0, small, small)], name="a.jpg")
    assert not result.accepted
    assert "terlalu kecil" in result.reason


def test_face_exactly_at_minimum_is_accepted():
    result = check_photo(noisy(), [box(0, 0, MIN_ENROLL_FACE, MIN_ENROLL_FACE)])
    assert result.accepted


def test_unreadable_image_is_rejected():
    result = check_photo(None, [box(0, 0, 200, 200)], name="rusak.jpg")
    assert not result.accepted
    assert "terbaca" in result.reason


# ---------- blur: peringatan, BUKAN penolakan ----------
def test_blurry_photo_is_warned_not_rejected():
    """Gerbang blur pernah menolak foto webcam yang sebenarnya sempurna.

    Diukur: embedding wajah bertahan sampai blur yang jauh lebih berat daripada
    yang bisa dibedakan metrik ini. Orang yang gagal mendaftar sama sekali tidak
    akan pernah dikenali kamera, jadi memblokir lebih merugikan daripada
    menerima foto yang kurang ideal.
    """
    result = check_photo(flat(), [box(0, 0, 200, 200)], name="buram.jpg")
    assert result.accepted, "foto buram tidak boleh DITOLAK"
    assert result.warnings, "tapi harus diperingatkan"
    assert "buram" in result.warnings[0]


def test_sharp_photo_has_no_warnings():
    result = check_photo(noisy(), [box(0, 0, 200, 200)], name="tajam.jpg")
    assert result.accepted
    assert result.warnings == []


def checkerboard(size: int = 400, cell: int = 40) -> np.ndarray:
    """Citra berstruktur — tepinya bertahan saat diperkecil, seperti wajah."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, cell):
        for x in range(0, size, cell):
            if ((x // cell) + (y // cell)) % 2 == 0:
                img[y:y + cell, x:x + cell] = 230
    return img


def raw_variance(img: np.ndarray) -> float:
    """Metrik lama: Laplacian langsung, tanpa normalisasi ukuran."""
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def test_sharpness_is_resolution_invariant():
    """Ketajaman yang sama harus memberi angka yang sama di resolusi berapa pun.

    Tanpa normalisasi, metriknya justru NAIK saat gambar diperkecil (tepi jadi
    lebih tajam relatif terhadap piksel), sehingga wajah yang jauh dari kamera
    tampak lebih tajam daripada wajah yang sama dari dekat — kebalikan dari
    yang masuk akal, dan membuat ambang apa pun bergantung pada jarak.
    """
    big = checkerboard(400, 40)
    ratios_raw, ratios_norm = [], []
    for size in (240, 120):
        small = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
        ratios_raw.append(raw_variance(small) / raw_variance(big))
        ratios_norm.append(
            face_sharpness(small, box(0, 0, size, size))
            / face_sharpness(big, box(0, 0, 400, 400))
        )

    # Metrik lama meleset jauh; yang ternormalisasi praktis tidak bergerak.
    assert max(ratios_raw) > 2.0, f"raw seharusnya bervariasi: {ratios_raw}"
    for ratio in ratios_norm:
        assert 0.9 < ratio < 1.1, f"ternormalisasi harus stabil: {ratios_norm}"


def test_flat_image_scores_near_zero():
    assert face_sharpness(flat(), box(0, 0, 200, 200)) < SOFT_SHARPNESS


# ---------- pendaftaran ----------
@pytest.fixture()
def book(tmp_path: Path) -> AttendanceBook:
    return AttendanceBook(db_path=tmp_path / "att.db", threshold=0.40)


def test_enroll_accepts_valid_photos(book: AttendanceBook):
    faces = [[box(0, 0, 200, 200)]] * 3
    detector = FakeDetector(faces)
    embedder = FakeEmbedder([unit(1), unit(2), unit(3)])
    photos = [(f"foto{i}.jpg", noisy(seed=i)) for i in range(3)]

    result = enroll_photos(book, detector, embedder, "EMP001", "Budi", "Produksi", photos)
    assert result.accepted == 3
    assert result.rejected == []
    assert book.list_employees()[0].num_embeddings == 3


def test_enroll_rejects_near_duplicate_photos(book: AttendanceBook):
    """Foto kembar tidak menambah cakupan pose apa pun."""
    same = unit(1)
    detector = FakeDetector([[box(0, 0, 200, 200)]] * 2)
    embedder = FakeEmbedder([same, same])
    photos = [("a.jpg", noisy(seed=1)), ("b.jpg", noisy(seed=2))]

    result = enroll_photos(book, detector, embedder, "EMP001", "Budi", "", photos)
    assert result.accepted == 1
    assert len(result.rejected) == 1
    assert "identik" in result.rejected[0].reason


def test_enroll_keeps_genuinely_different_poses(book: AttendanceBook):
    detector = FakeDetector([[box(0, 0, 200, 200)]] * 2)
    embedder = FakeEmbedder([unit(1), unit(2)])
    photos = [("a.jpg", noisy(seed=1)), ("b.jpg", noisy(seed=2))]

    result = enroll_photos(book, detector, embedder, "EMP001", "Budi", "", photos)
    assert result.accepted == 2


def test_employee_is_created_even_when_all_photos_fail(book: AttendanceBook):
    """Kegagalan harus terlihat di daftar, bukan menghilang tanpa jejak.

    Karyawan dengan 0 foto muncul dengan peringatan; karyawan yang tidak pernah
    dibuat sama sekali baru ditemukan berminggu-minggu kemudian sebagai "kok
    orang ini tidak pernah terbaca".
    """
    detector = FakeDetector([[]])
    embedder = FakeEmbedder()
    result = enroll_photos(book, detector, embedder, "EMP001", "Budi", "",
                           [("a.jpg", noisy())])

    assert result.accepted == 0
    employees = book.list_employees()
    assert len(employees) == 1
    assert employees[0].num_embeddings == 0


def test_enroll_result_serializes(book: AttendanceBook):
    detector = FakeDetector([[box(0, 0, 200, 200)], []])
    embedder = FakeEmbedder([unit(1)])
    photos = [("ok.jpg", noisy(seed=1)), ("gagal.jpg", noisy(seed=2))]

    payload = enroll_photos(book, detector, embedder, "EMP001", "Budi", "",
                            photos).to_dict()
    import json
    json.dumps(payload)
    assert payload["accepted"] == 1
    assert payload["rejected"][0]["file"] == "gagal.jpg"


# ---------- ketiga jalur memakai definisi yang sama ----------
def test_all_three_paths_share_one_definition():
    """CLI, UI, dan API harus memanggil `enroll_photos` yang sama.

    Kalau salah satu menuliskan aturannya sendiri lagi, karyawan yang
    didaftarkan lewat jalur itu diam-diam mendapat kualitas pendaftaran yang
    berbeda — dan tidak ada yang memberitahunya.
    """
    sources = {
        "CLI": PROJECT_ROOT / "scripts" / "enroll_faces.py",
        "UI": PROJECT_ROOT / "app" / "fatigue_ui.py",
        "API": PROJECT_ROOT / "app" / "fatigue_api.py",
    }
    for label, path in sources.items():
        text = path.read_text(encoding="utf-8")
        assert "enroll_photos" in text, f"{label} tidak memakai enroll_photos"
        assert "from src.fatigue.enrollment import" in text, \
            f"{label} tidak mengimpor aturan bersama"


def test_duplicate_threshold_is_stricter_than_identity_threshold():
    """Ambang duplikat harus jauh di atas ambang 'orang yang sama'.

    Kalau tidak, dua pose sah dari orang yang sama akan dibuang sebagai
    duplikat — persis foto-foto yang paling berguna untuk memperluas cakupan.
    """
    from src.fatigue.face import DEFAULT_THRESHOLDS

    assert DUPLICATE_SIM > DEFAULT_THRESHOLDS["sface"] * 2
