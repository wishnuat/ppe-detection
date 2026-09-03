"""Deteksi wajah (YuNet) + embedding wajah pluggable untuk absensi.

Pemisahan deteksi & embedding disengaja: absensi butuh vektor identitas,
sedangkan fatigue cuma butuh kotak wajah untuk di-crop ke landmarker. Dengan
dipisah, mode "fatigue tanpa absensi" tidak perlu memuat SFace 37 MB sama
sekali.

Dua backend embedding, dipilih lewat `EMBEDDER_BACKEND` / argumen:

    sface       cv2.FaceRecognizerSF (default). 128-d, 37 MB, nol dependency
                baru karena sudah ada di opencv-python. ~99.6% LFW.
    insightface buffalo_l / ArcFace R100. 512-d, lebih akurat pada pose ekstrem
                dan wajah kecil, tapi butuh `pip install insightface
                onnxruntime` dan unduhan ~330 MB saat pertama dipakai.

Keduanya menghasilkan vektor ter-L2-normalisasi, jadi kemiripan = dot product
dan `AttendanceBook` tidak perlu tahu backend mana yang aktif. Ambang default
berbeda per backend (skala similarity-nya memang tidak sama) — lihat
`DEFAULT_THRESHOLDS`.
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np

from src.fatigue.assets import ensure
from src.fatigue.types import FaceBox

EMBEDDER_BACKENDS = ("sface", "insightface")

# Ambang cosine similarity minimum untuk menyatakan "ini orang yang sama".
# Angka SFace mengikuti rekomendasi OpenCV Zoo (0.363 pada protokol LFW);
# dinaikkan ke 0.40 karena absensi lebih menderita akibat salah-orang
# (false accept) daripada akibat diminta menghadap kamera sekali lagi.
DEFAULT_THRESHOLDS = {"sface": 0.40, "insightface": 0.35}


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else vec


class FaceDetector:
    """Pembungkus cv2.FaceDetectorYN.

    YuNet menuntut input size di-set sama persis dengan ukuran frame sebelum
    `detect()`. Frame CCTV bisa berganti resolusi di tengah sesi (ganti kamera,
    ganti stream), jadi ukuran dicek tiap panggilan — biayanya nol kalau sama.

    Deteksi dijalankan pada frame yang diperkecil ke `detect_width` kalau frame
    aslinya lebih lebar. Biaya YuNet tumbuh dengan jumlah piksel, dan pada CPU
    uji ia 34 ms di 640px tapi 14 ms di 320px — untuk CCTV 1080p selisihnya
    jauh lebih besar lagi. Koordinat hasilnya diskalakan kembali ke frame asli,
    jadi landmarker dan embedder tetap bekerja pada resolusi penuh dan tidak
    kehilangan detail apa pun; yang berkurang hanya kemampuan menemukan wajah
    yang sangat kecil, dan itu dikompensasi dengan menskalakan `min_face` ikut
    turun.
    """

    def __init__(
        self,
        conf: float = 0.6,
        nms: float = 0.3,
        top_k: int = 50,
        min_face: int = 40,
        model_path: str | Path | None = None,
        detect_width: int | None = 640,
    ) -> None:
        self.conf = conf
        self.min_face = min_face
        self.detect_width = detect_width
        self.model_path = str(model_path or ensure("yunet", quiet=True))
        self._input_size = (320, 320)
        self._net = cv2.FaceDetectorYN.create(
            self.model_path, "", self._input_size, conf, nms, top_k
        )

    def _detection_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float]:
        """(frame untuk deteksi, faktor skala menuju koordinat asli)."""
        h, w = frame.shape[:2]
        if not self.detect_width or w <= self.detect_width:
            return frame, 1.0
        scale = self.detect_width / w
        resized = cv2.resize(
            frame, (self.detect_width, max(1, int(round(h * scale)))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, 1.0 / scale

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        h, w = frame.shape[:2]
        small, scale = self._detection_frame(frame)
        size = (small.shape[1], small.shape[0])
        if size != self._input_size:
            self._input_size = size
            self._net.setInputSize(size)

        _, faces = self._net.detect(small)
        if faces is None:
            return []

        # min_face dinyatakan dalam piksel frame ASLI, jadi saat deteksi
        # berjalan di frame yang diperkecil, ambangnya harus ikut mengecil.
        min_face_scaled = self.min_face / scale

        out: list[FaceBox] = []
        for row in faces:
            x, y, bw, bh = row[:4]
            if min(bw, bh) < min_face_scaled:
                # Wajah lebih kecil dari ini tidak menyisakan cukup piksel mata
                # untuk PERCLOS; melaporkannya cuma menghasilkan sinyal palsu.
                continue
            # Kolom 0..13 adalah koordinat (bbox + 5 landmark); kolom 14 skor.
            # Skor TIDAK boleh ikut diskalakan.
            scaled = [float(v) * scale for v in row[:14]] + [float(row[14])]
            x1, y1 = max(0, int(scaled[0])), max(0, int(scaled[1]))
            x2 = min(w, int(scaled[0] + scaled[2]))
            y2 = min(h, int(scaled[1] + scaled[3]))
            lms = [[int(scaled[4 + 2 * i]), int(scaled[5 + 2 * i])] for i in range(5)]
            out.append(
                FaceBox(
                    bbox=[x1, y1, x2, y2],
                    confidence=float(row[14]),
                    landmarks=lms,
                    # `raw` dipakai apa adanya oleh SFace.alignCrop pada frame
                    # resolusi PENUH, jadi yang disimpan harus versi yang sudah
                    # diskalakan — bukan keluaran mentah dari frame kecil.
                    raw=scaled,
                )
            )
        out.sort(key=lambda f: f.area, reverse=True)
        return out


class FaceEmbedder:
    """Antarmuka embedding wajah. Subclass mengisi `_embed`."""

    backend: str = "base"
    dim: int = 0

    @property
    def threshold(self) -> float:
        return DEFAULT_THRESHOLDS.get(self.backend, 0.4)

    def embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Vektor identitas ter-L2-normalisasi untuk satu wajah."""
        return _l2_normalize(self._embed(frame, face))

    def embed_many(self, frame: np.ndarray, faces: list[FaceBox]) -> list[np.ndarray]:
        return [self.embed(frame, f) for f in faces]

    def _embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        raise NotImplementedError


class SFaceEmbedder(FaceEmbedder):
    """cv2.FaceRecognizerSF — 128-d, jalan di atas opencv yang sudah terpasang."""

    backend = "sface"
    dim = 128

    def __init__(self, model_path: str | Path | None = None) -> None:
        self.model_path = str(model_path or ensure("sface", quiet=True))
        self._rec = cv2.FaceRecognizerSF.create(self.model_path, "")

    def _embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        if not face.raw:
            raise ValueError(
                "SFace butuh 5 landmark dari YuNet untuk alignment. "
                "FaceBox ini datang dari detektor lain — pakai backend "
                "insightface, atau deteksi ulang dengan FaceDetector."
            )
        row = np.asarray(face.raw, dtype=np.float32).reshape(1, -1)
        aligned = self._rec.alignCrop(frame, row)
        return self._rec.feature(aligned)


class InsightFaceEmbedder(FaceEmbedder):
    """ArcFace (buffalo_l) lewat paket `insightface`.

    Modelnya melakukan deteksi sendiri, jadi wajah dicari ulang di dalam crop
    (dengan margin) alih-alih memakai landmark YuNet — ArcFace sensitif pada
    alignment dan pakai template 5-titik sendiri yang berbeda dari YuNet.
    """

    backend = "insightface"
    dim = 512

    def __init__(self, model_name: str = "buffalo_l", det_size: int = 320) -> None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as exc:  # pragma: no cover - jalur opsional
            raise ImportError(
                "Backend 'insightface' butuh paket tambahan:\n"
                "    pip install insightface onnxruntime\n"
                "Atau pakai backend default 'sface' yang tidak butuh apa-apa."
            ) from exc

        self.model_path = model_name
        self._app = FaceAnalysis(
            name=model_name, allowed_modules=["detection", "recognition"]
        )
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))

    def _embed(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        x1, y1, x2, y2 = face.bbox
        # Margin 25%: ArcFace dilatih pada crop yang menyertakan dahi & dagu,
        # sedangkan box YuNet ketat di wajah saja.
        mw, mh = int((x2 - x1) * 0.25), int((y2 - y1) * 0.25)
        h, w = frame.shape[:2]
        crop = frame[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)]
        if crop.size == 0:
            return np.zeros(self.dim, dtype=np.float32)

        found = self._app.get(crop)
        if not found:
            return np.zeros(self.dim, dtype=np.float32)
        best = max(
            found, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        return np.asarray(best.normed_embedding, dtype=np.float32)


def build_embedder(backend: str | None = None, **kwargs) -> FaceEmbedder:
    """Bangun embedder sesuai nama backend (default dari env EMBEDDER_BACKEND)."""
    name = (backend or os.getenv("EMBEDDER_BACKEND", "sface")).strip().lower()
    if name not in EMBEDDER_BACKENDS:
        raise ValueError(
            f"Embedder '{backend}' tidak dikenal. "
            f"Pilihan: {', '.join(EMBEDDER_BACKENDS)}"
        )
    if name == "insightface":
        return InsightFaceEmbedder(**kwargs)
    return SFaceEmbedder(**kwargs)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Kemiripan dua vektor yang SUDAH dinormalisasi. Dipotong ke [-1, 1]."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    return float(np.clip(np.dot(a, b), -1.0, 1.0))
