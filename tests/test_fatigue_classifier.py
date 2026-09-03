"""Test classifier fatigue: preprocessing, kontrak checkpoint, dan urutan kelas.

Test terpenting di file ini adalah `test_training_and_inference_preprocessing_match`.
Skew antara transform training dan transform runtime adalah cara paling umum
sebuah classifier terlihat bagus di notebook lalu gagal di lapangan, dan ia
tidak memunculkan error apa pun — cuma akurasi yang diam-diam lebih rendah.
Satu-satunya cara mencegahnya adalah menguji bahwa keduanya menghasilkan tensor
yang sama, bukan sekadar berjanji di komentar.

Test di sini tidak butuh checkpoint terlatih; yang diuji adalah kontraknya.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.fatigue.classifier import (
    CLASSES,
    FATIGUE_INDEX,
    IMAGE_SIZE,
    MEAN,
    STD,
    build_classifier,
    preprocess_bgr,
    softmax,
)


def random_crop(seed: int = 0, h: int = 130, w: int = 90) -> np.ndarray:
    """Crop wajah palsu: BGR uint8 dengan ukuran yang tidak persegi."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)


# ---------- preprocessing ----------
def test_preprocess_shape_and_dtype():
    tensor = preprocess_bgr(random_crop())
    assert tensor.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    assert tensor.dtype == np.float32


def test_preprocess_applies_imagenet_normalization():
    """Piksel putih penuh harus mendarat di (1 - mean) / std per kanal."""
    white = np.full((50, 50, 3), 255, dtype=np.uint8)
    tensor = preprocess_bgr(white)
    expected = (1.0 - MEAN) / STD
    for channel in range(3):
        assert tensor[channel].mean() == pytest.approx(expected[channel], abs=1e-4)


def test_preprocess_converts_bgr_to_rgb():
    """Model dilatih pada RGB; melewatkan BGR apa adanya menukar dua kanal."""
    # Biru murni dalam BGR = (255, 0, 0).
    blue_bgr = np.zeros((40, 40, 3), dtype=np.uint8)
    blue_bgr[:, :, 0] = 255
    tensor = preprocess_bgr(blue_bgr)
    # Setelah konversi, kanal RGB ke-2 (biru) yang harus tinggi, bukan ke-0.
    assert tensor[2].mean() > tensor[0].mean()
    assert tensor[2].mean() == pytest.approx((1.0 - MEAN[2]) / STD[2], abs=1e-4)


def test_preprocess_accepts_grayscale():
    """Frame CCTV inframerah kadang datang sebagai satu kanal."""
    gray = np.full((60, 60), 128, dtype=np.uint8)
    assert preprocess_bgr(gray).shape == (3, IMAGE_SIZE, IMAGE_SIZE)


def test_preprocess_resizes_without_preserving_aspect():
    """Crop lonjong harus jadi persegi — sama seperti Resize((s, s)) di training."""
    tensor = preprocess_bgr(random_crop(h=200, w=50))
    assert tensor.shape[1] == tensor.shape[2] == IMAGE_SIZE


def test_training_and_inference_preprocessing_match():
    """Transform eval di training harus menghasilkan tensor yang sama persis.

    Kalau test ini gagal, salah satu dari dua jalur telah diubah tanpa yang
    lain — dan model akan melihat distribusi yang berbeda dari yang ia latih.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from PIL import Image

    from scripts.train_fatigue import build_transforms

    _, transform_eval = build_transforms(IMAGE_SIZE)
    crop = random_crop(seed=7, h=137, w=101)

    ours = preprocess_bgr(crop)
    # Jalur training membaca lewat PIL, yang bekerja di ruang RGB.
    theirs = transform_eval(Image.fromarray(crop[:, :, ::-1])).numpy()

    assert ours.shape == theirs.shape
    # Toleransi longgar karena resampling PIL dan cv2 tidak identik bit-per-bit;
    # yang diuji adalah tidak adanya perbedaan SISTEMATIS (kanal tertukar,
    # normalisasi berbeda, skala salah), bukan kesetaraan piksel sempurna.
    assert np.abs(ours.mean() - theirs.mean()) < 0.05
    assert np.abs(ours.std() - theirs.std()) < 0.05
    assert np.corrcoef(ours.ravel(), theirs.ravel())[0, 1] > 0.97


# ---------- softmax ----------
def test_softmax_sums_to_one():
    logits = np.array([[2.0, -1.0], [0.0, 0.0], [-5.0, 5.0]], dtype=np.float32)
    assert np.allclose(softmax(logits).sum(axis=-1), 1.0)


def test_softmax_is_numerically_stable():
    """Logit besar tidak boleh menghasilkan inf/nan."""
    logits = np.array([[1000.0, 999.0]], dtype=np.float32)
    probs = softmax(logits)
    assert np.isfinite(probs).all()
    assert probs.sum() == pytest.approx(1.0)


# ---------- kontrak kelas ----------
def test_class_order_is_locked():
    """Membalik urutan ini menukar arti 'lelah' dan 'segar' di seluruh sistem.

    Test ini ada supaya perubahan itu gagal di sini, bukan di lapangan —
    tidak satu pun metrik akurasi yang akan menunjukkan gejalanya.
    """
    assert CLASSES == ("nonfatigue", "fatigue")
    assert FATIGUE_INDEX == 1
    assert CLASSES[FATIGUE_INDEX] == "fatigue"


def test_dataset_prep_uses_the_same_class_order():
    from scripts.prepare_fatigue_dataset import CLASSES as PREP_CLASSES

    assert tuple(PREP_CLASSES) == CLASSES


# ---------- pemilihan backend ----------
def test_unknown_backend_is_rejected():
    with pytest.raises(ValueError, match="tidak dikenal"):
        build_classifier("tensorflow")


def test_missing_checkpoint_explains_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        build_classifier("torch", checkpoint=tmp_path / "tidak_ada.pt")
    message = str(exc.value)
    assert "train_fatigue.py" in message
    assert "prepare_fatigue_dataset.py" in message


# ---------- integrasi checkpoint (dilewati kalau belum dilatih) ----------
def test_trained_checkpoint_round_trips():
    """Checkpoint hasil training harus bisa dimuat dan memberi probabilitas valid."""
    pytest.importorskip("torch")
    from src.fatigue.classifier import DEFAULT_CHECKPOINT

    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Checkpoint belum ada — jalankan scripts/train_fatigue.py")

    classifier = build_classifier("torch")
    probs = classifier.predict_batch([random_crop(1), random_crop(2)])

    assert probs.shape == (2,)
    assert ((probs >= 0.0) & (probs <= 1.0)).all()
    # Ambang harus datang dari tuning di validation set, bukan 0.5 asal.
    assert 0.0 < classifier.threshold < 1.0
    assert classifier.metadata.get("classes") == list(CLASSES)


def test_tta_does_not_change_output_range():
    pytest.importorskip("torch")
    from src.fatigue.classifier import DEFAULT_CHECKPOINT

    if not DEFAULT_CHECKPOINT.exists():
        pytest.skip("Checkpoint belum ada")

    crops = [random_crop(3)]
    plain = build_classifier("torch", tta=False).predict_batch(crops)
    flipped = build_classifier("torch", tta=True).predict_batch(crops)
    assert 0.0 <= float(flipped[0]) <= 1.0
    # Wajah kurang lebih simetris, jadi TTA flip tidak boleh mengubah
    # keputusannya secara drastis pada input yang sama.
    assert abs(float(plain[0]) - float(flipped[0])) < 0.5
