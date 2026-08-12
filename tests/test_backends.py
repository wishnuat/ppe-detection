"""Unit test untuk pemilihan backend & pipeline pre/post-processing OpenVINO.

Semuanya berjalan tanpa file weights maupun runtime OpenVINO, jadi test tetap
hijau di CI yang tidak punya model — bagian yang butuh model asli sudah
diverifikasi terpisah lewat `scripts/benchmark.py` dan `scripts/evaluate.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends import BACKENDS, build_detector, normalize
from src.openvino_detector import letterbox, nms


# ---------- pemilihan backend ----------

def test_normalize_accepts_canonical_names():
    for name in BACKENDS:
        assert normalize(name) == name


def test_normalize_maps_legacy_aliases():
    # README & script lama memakai `--backend local`.
    assert normalize("local") == "torch"
    assert normalize("pytorch") == "torch"
    assert normalize("int8") == "openvino-int8"


def test_normalize_is_case_insensitive():
    assert normalize("OpenVINO") == "openvino"
    assert normalize("  TORCH ") == "torch"


def test_normalize_rejects_unknown_backend():
    with pytest.raises(ValueError, match="tidak dikenal"):
        normalize("tensorrt")


def test_normalize_defaults_to_torch(monkeypatch):
    monkeypatch.delenv("INFERENCE_BACKEND", raising=False)
    assert normalize(None) == "torch"


def test_normalize_reads_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_BACKEND", "openvino-int8")
    assert normalize(None) == "openvino-int8"


def test_build_detector_reports_missing_openvino_model(monkeypatch, tmp_path):
    """Pesan error harus menyebut cara memperbaikinya, bukan sekadar 'not found'."""
    monkeypatch.setenv("OPENVINO_MODEL_DIR", str(tmp_path / "tidak_ada"))
    with pytest.raises(FileNotFoundError, match="export_openvino"):
        build_detector("openvino")


# ---------- letterbox ----------

@pytest.mark.parametrize("h,w", [(480, 640), (640, 480), (720, 1280), (100, 100), (37, 911)])
def test_letterbox_output_is_exactly_target_size(h, w):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    out, _, _ = letterbox(img, (320, 320))
    assert out.shape == (320, 320, 3)


def test_letterbox_preserves_aspect_ratio():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    out, ratio, (pad_x, pad_y) = letterbox(img, (320, 320))
    assert ratio == pytest.approx(0.5)
    # 640*0.5 = 320 -> tidak ada pad horizontal; 360*0.5 = 180 -> pad vertikal.
    assert pad_x == 0
    assert pad_y == (320 - 180) // 2


def test_letterbox_roundtrip_maps_coordinates_back():
    """Titik di gambar asli harus kembali ke posisi semula setelah unpad."""
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    _, ratio, (pad_x, pad_y) = letterbox(img, (320, 320))
    for x, y in [(0, 0), (640, 360), (1279, 719)]:
        lx, ly = x * ratio + pad_x, y * ratio + pad_y
        assert (lx - pad_x) / ratio == pytest.approx(x, abs=1.0)
        assert (ly - pad_y) / ratio == pytest.approx(y, abs=1.0)


def test_letterbox_pads_with_yolo_gray():
    img = np.full((100, 300, 3), 255, dtype=np.uint8)
    out, _, _ = letterbox(img, (320, 320))
    # Baris paling atas pasti area padding untuk gambar yang sangat lebar.
    assert tuple(out[0, 0]) == (114, 114, 114)


# ---------- NMS ----------

def test_nms_keeps_single_box():
    boxes = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert nms(boxes, np.array([0.9]), 0.45) == [0]


def test_nms_suppresses_overlapping_lower_score():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32)
    keep = nms(boxes, np.array([0.9, 0.8]), 0.45)
    assert keep == [0]


def test_nms_keeps_disjoint_boxes():
    boxes = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    keep = nms(boxes, np.array([0.7, 0.9]), 0.45)
    assert sorted(keep) == [0, 1]


def test_nms_returns_indices_in_score_order():
    boxes = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [200, 200, 210, 210]],
                     dtype=np.float32)
    assert nms(boxes, np.array([0.3, 0.9, 0.6]), 0.45) == [1, 2, 0]


def test_nms_handles_empty_input():
    assert nms(np.zeros((0, 4), dtype=np.float32), np.array([]), 0.45) == []


def test_nms_threshold_controls_suppression():
    # IoU dua box ini ~0.68: tersimpan pada threshold longgar, tertekan pada ketat.
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 15]], dtype=np.float32)
    assert len(nms(boxes, np.array([0.9, 0.8]), 0.9)) == 2
    assert len(nms(boxes, np.array([0.9, 0.8]), 0.5)) == 1
