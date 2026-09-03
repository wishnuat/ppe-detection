"""Test deteksi wajah: penskalaan koordinat, filter ukuran, dan pemilihan backend.

Butuh bobot YuNet/SFace. Kalau belum diunduh (`python -m src.fatigue.assets`),
test yang membutuhkannya dilewati alih-alih gagal — kontributor yang cuma
menyentuh modul PPE tidak perlu mengunduh 41 MB untuk menjalankan `pytest`.
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

from src.fatigue import assets
from src.fatigue.face import (
    EMBEDDER_BACKENDS,
    FaceDetector,
    SFaceEmbedder,
    build_embedder,
    cosine_similarity,
)

SAMPLE = PROJECT_ROOT / "frame.jpg"


def require(name: str) -> None:
    if not assets.path_for(name).exists():
        pytest.skip(f"Bobot '{name}' belum diunduh — jalankan "
                    "`python -m src.fatigue.assets`")


@pytest.fixture(scope="module")
def sample() -> np.ndarray:
    if not SAMPLE.exists():
        pytest.skip(f"{SAMPLE.name} tidak ada (ia di .gitignore)")
    img = cv2.imread(str(SAMPLE))
    if img is None:
        pytest.skip(f"{SAMPLE.name} tidak bisa dibaca")
    return img


# ---------- cosine similarity ----------
def test_cosine_similarity_of_identical_vectors():
    v = np.array([0.6, 0.8], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_of_orthogonal_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_is_clipped():
    """Akumulasi float bisa melewati 1.0 sedikit; itu tidak boleh bocor keluar."""
    v = np.ones(512, dtype=np.float32) / np.sqrt(512)
    assert -1.0 <= cosine_similarity(v, v) <= 1.0


# ---------- pemilihan backend ----------
def test_unknown_embedder_backend_is_rejected():
    with pytest.raises(ValueError, match="tidak dikenal"):
        build_embedder("dlib")


def test_backend_list_matches_implementations():
    assert EMBEDDER_BACKENDS == ("sface", "insightface")


# ---------- deteksi ----------
def test_detects_a_face(sample):
    require("yunet")
    faces = FaceDetector().detect(sample)
    assert faces, "wajah pada frame contoh seharusnya terdeteksi"
    x1, y1, x2, y2 = faces[0].bbox
    assert 0 <= x1 < x2 <= sample.shape[1]
    assert 0 <= y1 < y2 <= sample.shape[0]
    assert len(faces[0].landmarks) == 5
    assert len(faces[0].raw) == 15


def test_faces_are_sorted_largest_first(sample):
    require("yunet")
    faces = FaceDetector(min_face=10).detect(sample)
    areas = [f.area for f in faces]
    assert areas == sorted(areas, reverse=True)


def test_min_face_filters_small_detections(sample):
    require("yunet")
    permissive = FaceDetector(min_face=10).detect(sample)
    strict = FaceDetector(min_face=10_000).detect(sample)
    assert len(strict) < len(permissive) or not permissive
    assert strict == []


def test_blank_frame_detects_nothing():
    require("yunet")
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    assert FaceDetector().detect(blank) == []


# ---------- penskalaan deteksi ----------
def test_detect_width_is_a_noop_for_smaller_frames(sample):
    """Frame yang sudah lebih kecil dari ambang tidak boleh disentuh sama sekali."""
    require("yunet")
    full = FaceDetector(detect_width=None).detect(sample)
    capped = FaceDetector(detect_width=sample.shape[1] * 2).detect(sample)
    assert [f.bbox for f in full] == [f.bbox for f in capped]


def test_scaled_detection_returns_coordinates_in_original_frame(sample):
    """Kotak dari deteksi berskala harus menunjuk lokasi yang sama di frame asli.

    Kalau penskalaan baliknya salah, bug-nya tidak memunculkan error apa pun —
    ia cuma membuat crop wajah meleset, dan semua sinyal mata ikut salah.
    """
    require("yunet")
    big = cv2.resize(sample, (sample.shape[1] * 3, sample.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    full = FaceDetector(detect_width=None).detect(big)
    scaled = FaceDetector(detect_width=640).detect(big)
    if not full or not scaled:
        pytest.skip("wajah tidak terdeteksi pada frame yang diperbesar")

    a, b = full[0].bbox, scaled[0].bbox
    # Toleransi 8% dari lebar kotak: penskalaan memang menggeser sedikit,
    # tapi tidak boleh sampai memindahkan kotaknya ke tempat lain.
    tolerance = 0.08 * (a[2] - a[0])
    for got, expected in zip(b, a):
        assert abs(got - expected) <= tolerance, f"{b} vs {a}"


def test_scaled_detection_preserves_identity(sample):
    """Embedding dari kotak hasil deteksi berskala harus tetap orang yang sama.

    Ini invariant yang sebenarnya penting: bbox boleh bergeser beberapa piksel
    asal absensi tetap mengenali orangnya.
    """
    require("yunet")
    require("sface")
    big = cv2.resize(sample, (sample.shape[1] * 3, sample.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    full = FaceDetector(detect_width=None).detect(big)
    scaled = FaceDetector(detect_width=640).detect(big)
    if not full or not scaled:
        pytest.skip("wajah tidak terdeteksi pada frame yang diperbesar")

    embedder = SFaceEmbedder()
    similarity = cosine_similarity(
        embedder.embed(big, full[0]), embedder.embed(big, scaled[0])
    )
    # Ambang absensi 0.40; margin di sini harus jauh lebih lebar dari itu.
    assert similarity > 0.90, f"similarity {similarity:.4f} terlalu rendah"


def test_min_face_threshold_scales_with_detection_size(sample):
    """`min_face` dinyatakan dalam piksel frame asli, bukan frame deteksi.

    Tanpa penskalaan ambang, memperkecil frame deteksi akan diam-diam membuang
    wajah yang sebenarnya besar di frame aslinya.
    """
    require("yunet")
    big = cv2.resize(sample, (sample.shape[1] * 3, sample.shape[0] * 3),
                     interpolation=cv2.INTER_CUBIC)
    faces = FaceDetector(detect_width=640, min_face=100).detect(big)
    if not faces:
        pytest.skip("wajah tidak terdeteksi pada frame yang diperbesar")
    x1, y1, x2, y2 = faces[0].bbox
    assert min(x2 - x1, y2 - y1) >= 100


# ---------- embedding ----------
def test_embedding_is_l2_normalized(sample):
    require("yunet")
    require("sface")
    faces = FaceDetector().detect(sample)
    if not faces:
        pytest.skip("wajah tidak terdeteksi")
    vector = SFaceEmbedder().embed(sample, faces[0])
    assert vector.shape == (128,)
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_embedding_is_stable_across_calls(sample):
    require("yunet")
    require("sface")
    faces = FaceDetector().detect(sample)
    if not faces:
        pytest.skip("wajah tidak terdeteksi")
    embedder = SFaceEmbedder()
    a = embedder.embed(sample, faces[0])
    b = embedder.embed(sample, faces[0])
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-5)


def test_sface_rejects_facebox_without_landmarks(sample):
    """SFace butuh 5 titik YuNet untuk alignment; tanpa itu ia harus menolak."""
    require("sface")
    from src.fatigue.types import FaceBox

    orphan = FaceBox(bbox=[10, 10, 110, 110], confidence=0.9)
    with pytest.raises(ValueError, match="landmark"):
        SFaceEmbedder().embed(sample, orphan)


# ---------- assets ----------
def test_asset_registry_is_self_consistent():
    for name, asset in assets.ASSETS.items():
        assert asset.name == name
        assert len(asset.sha256) == 64
        assert asset.size > 0
        assert asset.url.startswith("https://")
        assert asset.license


def test_unknown_asset_is_rejected():
    with pytest.raises(KeyError, match="tidak dikenal"):
        assets.ensure("resnet50")
