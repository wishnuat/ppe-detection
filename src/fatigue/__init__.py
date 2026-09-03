"""Deteksi fatigue pekerja + absensi face recognition dari CCTV.

Sub-paket ini berdiri sendiri dari `src.detector` (PPE): keduanya membaca frame
yang sama tapi menjawab pertanyaan berbeda, punya model, ambang, dan siklus
hidup sendiri. Yang menyatukan keduanya hanya `app/` (API & UI).

Alur runtime satu frame:

    frame -> FaceDetector      : bbox + 5 landmark per wajah
          -> FaceEmbedder      : vektor 128-d -> AttendanceBook (siapa ini?)
          -> FaceLandmarker    : 478 landmark + blendshape -> FatigueSignals
          -> FatigueClassifier : skor CNN per-frame (dilatih dari dataset Kaggle)
          -> FatigueFusion     : gabung skor CNN + PERCLOS/yawn/nod jadi level
          -> FatigueMonitor    : state per orang, histeresis, event alert

Import berat (torch, mediapipe) sengaja ditunda sampai kelasnya dipakai supaya
`import src.fatigue` tetap murah untuk kode yang cuma butuh dataclass-nya.
"""
from __future__ import annotations

from src.fatigue.types import (
    FaceBox,
    FatigueLevel,
    FatigueSignals,
    FrameAnalysis,
    Identity,
    PersonState,
)

__all__ = [
    "FaceBox",
    "FatigueLevel",
    "FatigueSignals",
    "FrameAnalysis",
    "Identity",
    "PersonState",
]
