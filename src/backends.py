"""Pemilihan backend inference, dipakai bersama oleh CLI, FastAPI, dan Streamlit.

Semua backend mewarisi `PPEDetector`, jadi pemanggilnya tidak perlu tahu mana
yang aktif — cukup `build_detector(...)` lalu pakai `predict_frame`.

    torch      models/best.pt lewat Ultralytics/PyTorch. Referensi akurasi.
    openvino   models/best_openvino_model (FP32). Cepat di CPU/iGPU Intel.
    openvino-int8
               models/best_int8_openvino_model. Tercepat, mAP sedikit turun.
    roboflow   Serverless API. Butuh internet, dipakai sebagai baseline.
"""
from __future__ import annotations

import os

from src.detector import PPEDetector

BACKENDS = ("torch", "openvino", "openvino-int8", "roboflow")

# Alias lama: `local` dulu berarti satu-satunya backend offline (PyTorch).
_ALIASES = {"local": "torch", "pytorch": "torch", "ov": "openvino", "int8": "openvino-int8"}

BACKEND_LABELS = {
    "torch": "PyTorch (models/best.pt)",
    "openvino": "OpenVINO FP32 (Intel CPU/iGPU)",
    "openvino-int8": "OpenVINO INT8 (terkuantisasi)",
    "roboflow": "Roboflow Serverless (online)",
}


def normalize(backend: str | None) -> str:
    name = (backend or os.getenv("INFERENCE_BACKEND", "torch")).strip().lower()
    name = _ALIASES.get(name, name)
    if name not in BACKENDS:
        raise ValueError(
            f"Backend '{backend}' tidak dikenal. Pilihan: {', '.join(BACKENDS)}"
        )
    return name


def build_detector(
    backend: str | None = None,
    conf: float | None = None,
    iou: float | None = None,
    device: str | None = None,
) -> PPEDetector:
    """Buat detector sesuai backend. `device` hanya dipakai backend OpenVINO."""
    name = normalize(backend)

    if name == "roboflow":
        from src.remote import RoboflowDetector

        return RoboflowDetector(conf=conf, iou=iou)

    if name.startswith("openvino"):
        from src.openvino_detector import OpenVINODetector

        return OpenVINODetector(
            device=device, conf=conf, iou=iou, int8=name.endswith("int8")
        )

    return PPEDetector(conf=conf, iou=iou)


def describe(detector: PPEDetector) -> str:
    """Satu baris ringkas tentang backend yang aktif, untuk log & UI."""
    device = getattr(detector, "device", None)
    suffix = f" @ {device}" if device else ""
    return f"{detector.model_path}{suffix} ({len(detector.class_names)} kelas)"
