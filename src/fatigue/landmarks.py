"""Sinyal perilaku fatigue dari landmark wajah (MediaPipe FaceLandmarker).

Modul ini menjawab satu pertanyaan per frame: "mata orang ini sedang terbuka
atau tertutup, mulutnya menganga atau tidak, kepalanya menunduk atau tidak?"
Agregasi antar-waktu (PERCLOS, laju kedip, microsleep) bukan urusan di sini —
itu ada di `temporal.py`.

Dua sumber sinyal dipakai bersamaan, sengaja redundan:

    Blendshape  `eyeBlinkLeft/Right`, `jawOpen` — keluaran kepala regresi
                MediaPipe, sudah dinormalisasi antar-orang dan tahan terhadap
                bentuk mata sipit/lebar. Ini sumber utama.
    Geometri    EAR & MAR dari koordinat landmark. Absolutnya tidak
                sebanding antar-orang, tapi *perubahannya* relatif terhadap
                baseline orang itu sangat informatif, dan ia tetap ada ketika
                blendshape gagal. Ini cadangan + bahan kalibrasi.

Landmarker dijalankan pada crop wajah, bukan frame penuh: di CCTV, wajah bisa
cuma 5% dari frame, dan menjalankan FaceMesh di resolusi penuh membuang waktu
sekaligus menurunkan presisi landmark mata.
"""
from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
import numpy as np

from src.fatigue.assets import ensure
from src.fatigue.types import FaceBox, FatigueSignals

# Indeks landmark FaceMesh (topologi 478 titik).
# Urutan tiap mata: [sudut luar, atas-1, atas-2, sudut dalam, bawah-2, bawah-1]
# — persis urutan yang diasumsikan rumus EAR Soukupova & Cech (2016).
LEFT_EYE = (362, 385, 387, 263, 373, 380)
RIGHT_EYE = (33, 160, 158, 133, 153, 144)
# Mulut: dua sudut + tiga pasang atas/bawah, mengikuti pola rumus yang sama.
MOUTH_CORNERS = (61, 291)
MOUTH_PAIRS = ((81, 178), (13, 14), (311, 402))

# Ambang default. Semuanya bisa dioverride per orang setelah kalibrasi
# (lihat `temporal.PersonTracker`), jadi angka di sini hanya titik awal.
DEFAULT_BLINK_THRESHOLD = 0.45   # blendshape eyeBlink: >= ini dianggap tertutup
DEFAULT_EAR_THRESHOLD = 0.21     # EAR geometris, cadangan saat blendshape absen
DEFAULT_MAR_THRESHOLD = 0.60     # MAR: >= ini mulut menganga (kandidat menguap)
DEFAULT_JAW_THRESHOLD = 0.40     # blendshape jawOpen
# Menunduk lebih dari ini dianggap kepala terkulai, bukan sekadar melihat meja.
DEFAULT_NOD_PITCH = 22.0


def _aspect_ratio(pts: np.ndarray, idx: tuple[int, ...]) -> float:
    """Rasio (tinggi rata-rata / lebar) untuk enam titik gaya EAR."""
    p1, p2, p3, p4, p5, p6 = (pts[i] for i in idx)
    width = float(np.linalg.norm(p1 - p4))
    if width < 1e-6:
        return 0.0
    height = float(np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5))
    return height / (2.0 * width)


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR rata-rata dua mata. ~0.30 saat terbuka, ~0.10 saat terpejam."""
    return 0.5 * (_aspect_ratio(pts, LEFT_EYE) + _aspect_ratio(pts, RIGHT_EYE))


def mouth_aspect_ratio(pts: np.ndarray) -> float:
    """MAR: tinggi bukaan mulut dibagi lebarnya. Menguap ~0.7+."""
    left, right = pts[MOUTH_CORNERS[0]], pts[MOUTH_CORNERS[1]]
    width = float(np.linalg.norm(left - right))
    if width < 1e-6:
        return 0.0
    height = sum(float(np.linalg.norm(pts[a] - pts[b])) for a, b in MOUTH_PAIRS)
    return height / (3.0 * width)


def head_pose_from_matrix(matrix: np.ndarray) -> tuple[float, float, float]:
    """Ekstrak (pitch, yaw, roll) derajat dari matriks transformasi 4x4.

    Konvensi keluaran: pitch positif = menunduk, yaw positif = menoleh kanan,
    roll positif = kepala miring ke kanan.
    """
    r = np.asarray(matrix, dtype=np.float64)[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy < 1e-6:
        # Gimbal lock: roll tidak bisa dipisahkan dari yaw, jadi dinolkan.
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    else:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


class FaceLandmarker:
    """Pembungkus MediaPipe FaceLandmarker yang aman dipanggil dari FastAPI.

    Objek Tasks MediaPipe tidak thread-safe, sedangkan Uvicorn menjalankan
    endpoint sync di threadpool — jadi tiap panggilan dikunci. Biayanya
    diabaikan dibanding inferensinya sendiri, dan alternatifnya (satu
    landmarker per thread) menggandakan memori tanpa untung nyata di CPU.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        margin: float = 0.25,
        min_detection_confidence: float = 0.4,
    ) -> None:
        # Import mediapipe ditunda sampai kelas ini benar-benar dibuat: ia
        # memuat runtime TFLite + XNNPACK yang butuh waktu beberapa ratus ms,
        # dan `import src.fatigue` tidak boleh membayar itu.
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.model_path = str(model_path or ensure("face_landmarker", quiet=True))
        self.margin = margin
        self._lock = threading.Lock()
        # Disimpan sebagai atribut supaya `analyze` tidak menyentuh sistem
        # import sama sekali di jalur panas.
        self._mp_image_cls = mp.Image
        self._mp_format = mp.ImageFormat

        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=self.model_path),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    # ---------- crop ----------
    def crop(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Potong wajah dengan margin. Kosong kalau box keluar frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = face.bbox
        mw, mh = int((x2 - x1) * self.margin), int((y2 - y1) * self.margin)
        return frame[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)]

    # ---------- inti ----------
    def analyze(self, frame: np.ndarray, face: FaceBox) -> FatigueSignals:
        """Sinyal perilaku untuk satu wajah. Semua field None kalau gagal."""
        crop = self.crop(frame, face)
        if crop.size == 0 or min(crop.shape[:2]) < 24:
            return FatigueSignals()

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        image = self._mp_image_cls(image_format=self._mp_format.SRGB, data=rgb)
        with self._lock:
            result = self._landmarker.detect(image)

        if not result.face_landmarks:
            return FatigueSignals()

        ch, cw = crop.shape[:2]
        pts = np.array(
            [[lm.x * cw, lm.y * ch] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        ear = eye_aspect_ratio(pts)
        mar = mouth_aspect_ratio(pts)

        blink = jaw = None
        if result.face_blendshapes:
            scores = {c.category_name: c.score for c in result.face_blendshapes[0]}
            left = scores.get("eyeBlinkLeft")
            right = scores.get("eyeBlinkRight")
            if left is not None and right is not None:
                # max, bukan rata-rata: satu mata yang jelas terpejam sudah
                # cukup menandakan penutupan — merata-rata dengan mata yang
                # sedang tertutup sebagian justru meredam sinyalnya.
                blink = float(max(left, right))
            jaw = scores.get("jawOpen")
            if jaw is not None:
                jaw = float(jaw)

        pitch = yaw = roll = None
        if result.facial_transformation_matrixes:
            pitch, yaw, roll = head_pose_from_matrix(
                result.facial_transformation_matrixes[0]
            )

        eye_closed = (
            blink >= DEFAULT_BLINK_THRESHOLD if blink is not None
            else ear <= DEFAULT_EAR_THRESHOLD
        )
        mouth_open = (
            jaw >= DEFAULT_JAW_THRESHOLD if jaw is not None
            else mar >= DEFAULT_MAR_THRESHOLD
        )

        return FatigueSignals(
            ear=float(ear),
            mar=float(mar),
            eye_closed=bool(eye_closed),
            mouth_open=bool(mouth_open),
            pitch=pitch,
            yaw=yaw,
            roll=roll,
            blink_score=blink,
            jaw_open=jaw,
        )

    def close(self) -> None:
        with self._lock:
            self._landmarker.close()
