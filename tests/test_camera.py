"""Test pembukaan webcam yang tahan-gantung.

`cv2.VideoCapture` bisa menggantung tanpa batas waktu di Windows. Terukur di
mesin uji: pembukaan pertama setelah jeda berhasil dalam 0,3 detik lewat
DirectShow, lalu dua pembukaan berikutnya tidak pernah kembali sama sekali —
dihentikan setelah 25 detik. Media Foundation pada saat yang sama tetap
membuka dalam 0,9 detik, berulang kali.

Gejalanya jahat karena tidak terlihat seperti error: program diam, tidak ada
window, tidak ada pesan. Test di sini memakai kamera palsu untuk memastikan
perilaku itu tidak kembali — tanpa perlu webcam sungguhan, jadi ia tetap jalan
di CI.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pytest

from src import camera as camera_module
from src.camera import BACKENDS, CameraOpenError, describe_camera, open_camera


class FakeCapture:
    def __init__(self, opened: bool = True, width: int = 640, height: int = 480,
                 fps: float = 30.0) -> None:
        self._opened = opened
        self._props = {
            cv2.CAP_PROP_FRAME_WIDTH: width,
            cv2.CAP_PROP_FRAME_HEIGHT: height,
            cv2.CAP_PROP_FPS: fps,
        }
        self.released = False

    def isOpened(self) -> bool:            # noqa: N802 - meniru API cv2
        return self._opened

    def get(self, prop):
        return self._props.get(prop, 0.0)

    def release(self) -> None:
        self.released = True


def fake_videocapture(behaviour: dict):
    """Bikin pengganti cv2.VideoCapture dengan perilaku per-backend.

    `behaviour` memetakan konstanta API ke salah satu: "ok", "gagal", atau
    "gantung" (tidur jauh lebih lama daripada timeout).
    """
    calls = []

    def factory(index, api):
        calls.append(api)
        mode = behaviour.get(api, "gagal")
        if mode == "gantung":
            time.sleep(30)
            return FakeCapture(opened=True)
        return FakeCapture(opened=(mode == "ok"))

    factory.calls = calls
    return factory


# ---------- urutan backend ----------
def test_media_foundation_is_tried_first():
    """MSMF didahulukan karena DirectShow yang terbukti menggantung.

    Selisih 0,6 detik saat lancar tidak sebanding dengan risiko menggantung
    selamanya.
    """
    assert BACKENDS[0][0] == cv2.CAP_MSMF
    assert cv2.CAP_DSHOW in [api for api, _ in BACKENDS]


def test_opens_with_first_working_backend(monkeypatch):
    factory = fake_videocapture({cv2.CAP_MSMF: "ok"})
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)

    cap, name = open_camera(0, timeout=1.0, warn=None)
    assert cap.isOpened()
    assert name == "Media Foundation"
    assert factory.calls == [cv2.CAP_MSMF]      # tidak mencoba yang lain


# ---------- inti: menggantung tidak boleh menggantungkan program ----------
def test_hanging_backend_falls_through_within_timeout(monkeypatch):
    """Backend yang menggantung harus ditinggalkan, bukan ditunggu selamanya."""
    factory = fake_videocapture({cv2.CAP_MSMF: "gantung", cv2.CAP_DSHOW: "ok"})
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)

    t0 = time.perf_counter()
    cap, name = open_camera(0, timeout=0.3, warn=None)
    elapsed = time.perf_counter() - t0

    assert name == "DirectShow"
    assert cap.isOpened()
    # Harus menyerah jauh sebelum 30 detik tidur milik backend palsu itu.
    assert elapsed < 5.0, f"butuh {elapsed:.1f}s — timeout tidak bekerja"


def test_all_backends_hanging_raises_instead_of_blocking(monkeypatch):
    factory = fake_videocapture({api: "gantung" for api, _ in BACKENDS})
    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)

    t0 = time.perf_counter()
    with pytest.raises(CameraOpenError):
        open_camera(0, timeout=0.2, warn=None)
    assert time.perf_counter() - t0 < 5.0


def test_error_message_is_actionable(monkeypatch):
    """Pesan gagal harus menyebut apa yang bisa dicoba, bukan sekadar 'gagal'."""
    monkeypatch.setattr(camera_module.cv2, "VideoCapture",
                        fake_videocapture({}))
    with pytest.raises(CameraOpenError) as exc:
        open_camera(3, timeout=0.2, warn=None)

    message = str(exc.value)
    assert "--index" in message
    assert "inframerah" in message          # laptop sering punya kamera IR
    assert "Zoom" in message or "Teams" in message
    assert "3" in message                   # index yang diminta ikut disebut


def test_failed_capture_is_released(monkeypatch):
    """Capture yang terbuka tapi tidak siap harus dilepas, bukan dibiarkan."""
    created = []

    def factory(index, api):
        cap = FakeCapture(opened=False)
        created.append(cap)
        return cap

    monkeypatch.setattr(camera_module.cv2, "VideoCapture", factory)
    with pytest.raises(CameraOpenError):
        open_camera(0, timeout=0.5, warn=None)
    assert created and all(c.released for c in created)


def test_warn_is_called_for_each_failed_backend(monkeypatch):
    monkeypatch.setattr(camera_module.cv2, "VideoCapture",
                        fake_videocapture({cv2.CAP_ANY: "ok"}))
    messages = []
    open_camera(0, timeout=0.3, warn=messages.append)
    # MSMF dan DSHOW gagal sebelum CAP_ANY berhasil.
    assert len(messages) == 2
    assert all("kamera" in m for m in messages)


# ---------- deskripsi ----------
def test_describe_camera_reports_resolution_and_fps():
    assert describe_camera(FakeCapture(width=1280, height=720, fps=30)) == "1280x720 @ 30 fps"


def test_describe_camera_omits_fps_when_unavailable():
    """Sebagian driver melaporkan FPS 0; menuliskan '@ 0 fps' cuma membingungkan."""
    assert describe_camera(FakeCapture(width=640, height=480, fps=0)) == "640x480"
