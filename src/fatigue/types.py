"""Tipe data bersama untuk pipeline fatigue & absensi.

Semua dataclass di sini murni data — tidak mengimpor torch, cv2, atau
mediapipe — supaya API, UI, dan test bisa memakainya tanpa memuat model.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum


class FatigueLevel(str, Enum):
    """Level kelelahan yang dilaporkan ke operator.

    Sengaja 4 tingkat, bukan biner: operator butuh peringatan dini (WASPADA)
    sebelum kondisi benar-benar berbahaya (KRITIS), dan butuh cara menyatakan
    "wajah tidak cukup terlihat untuk dinilai" (TIDAK_DIKETAHUI) yang berbeda
    dari "orangnya segar" — dua hal itu tidak boleh tertukar di dashboard.
    """

    UNKNOWN = "TIDAK_DIKETAHUI"
    ALERT = "SEGAR"
    MILD = "WASPADA"
    SEVERE = "LELAH"
    CRITICAL = "KRITIS"

    @property
    def severity(self) -> int:
        """Urutan numerik untuk perbandingan & sorting. UNKNOWN = -1."""
        return _SEVERITY[self]

    def __lt__(self, other: "FatigueLevel") -> bool:  # type: ignore[override]
        return self.severity < other.severity


_SEVERITY = {
    FatigueLevel.UNKNOWN: -1,
    FatigueLevel.ALERT: 0,
    FatigueLevel.MILD: 1,
    FatigueLevel.SEVERE: 2,
    FatigueLevel.CRITICAL: 3,
}

# Warna BGR untuk rendering, sejalan dengan konvensi src/detector.py.
LEVEL_COLORS = {
    FatigueLevel.UNKNOWN: (128, 128, 128),
    FatigueLevel.ALERT: (0, 200, 0),
    FatigueLevel.MILD: (0, 200, 255),
    FatigueLevel.SEVERE: (0, 90, 255),
    FatigueLevel.CRITICAL: (0, 0, 255),
}


@dataclass
class FaceBox:
    """Satu wajah terdeteksi pada satu frame."""

    bbox: list[int]              # [x1, y1, x2, y2] dalam piksel frame
    confidence: float
    # 5 titik YuNet: mata kanan, mata kiri, hidung, sudut mulut kanan/kiri.
    # Dipakai untuk alignment sebelum embedding; kosong kalau detektor lain.
    landmarks: list[list[int]] = field(default_factory=list)
    # Baris mentah YuNet (15 kolom). Disimpan karena `cv2.FaceRecognizerSF.
    # alignCrop` memintanya apa adanya; jangan direkonstruksi manual.
    raw: list[float] = field(default_factory=list)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return max(0, x2 - x1) * max(0, y2 - y1)

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "landmarks": self.landmarks,
        }


@dataclass
class Identity:
    """Hasil pencocokan wajah ke daftar karyawan terdaftar."""

    employee_id: str | None       # None = tidak dikenali
    name: str
    similarity: float             # cosine similarity 0..1 terhadap centroid
    is_known: bool

    @classmethod
    def unknown(cls, similarity: float = 0.0) -> "Identity":
        return cls(employee_id=None, name="Tidak dikenal",
                   similarity=similarity, is_known=False)

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "similarity": round(self.similarity, 4),
            "is_known": self.is_known,
        }


@dataclass
class FatigueSignals:
    """Sinyal perilaku mentah dari satu frame (belum diagregasi antar-waktu).

    Nilai None berarti sinyal itu tidak bisa diukur pada frame ini (wajah
    terlalu kecil, menoleh, atau landmark gagal) — berbeda dari nilai 0.
    """

    ear: float | None = None            # eye aspect ratio, rata-rata dua mata
    mar: float | None = None            # mouth aspect ratio (indikator menguap)
    eye_closed: bool = False
    mouth_open: bool = False
    # Sudut kepala derajat: pitch (angguk), yaw (menoleh), roll (miring).
    pitch: float | None = None
    yaw: float | None = None
    roll: float | None = None
    # Skor blendshape mediapipe (0..1) — lebih tahan terhadap variasi bentuk
    # mata antar-orang dibanding EAR geometris murni.
    blink_score: float | None = None
    jaw_open: float | None = None

    @property
    def usable(self) -> bool:
        return self.ear is not None

    def to_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


@dataclass
class PersonState:
    """Kondisi fatigue satu orang setelah agregasi temporal.

    Ini yang ditampilkan di dashboard; `FatigueSignals` adalah bahan mentahnya.
    """

    identity: Identity
    level: FatigueLevel = FatigueLevel.UNKNOWN
    score: float = 0.0                  # 0..1, skor fusi akhir
    cnn_score: float = 0.0              # 0..1, probabilitas kelas "fatigue"
    perclos: float = 0.0                # fraksi waktu mata tertutup di jendela
    blink_rate: float = 0.0             # kedipan per menit
    yawn_rate: float = 0.0              # menguap per menit
    nod_rate: float = 0.0               # kepala terkulai per menit
    microsleep_count: int = 0           # mata tertutup >= MICROSLEEP_SECONDS
    longest_closure: float = 0.0        # detik, penutupan mata terpanjang
    observed_seconds: float = 0.0       # lama pengamatan efektif
    reasons: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["identity"] = self.identity.to_dict()
        d["level"] = self.level.value
        for k in ("score", "cnn_score", "perclos", "blink_rate", "yawn_rate",
                  "nod_rate", "longest_closure", "observed_seconds"):
            d[k] = round(d[k], 4)
        return d


@dataclass
class FrameAnalysis:
    """Hasil lengkap satu frame: siapa saja yang terlihat & kondisinya."""

    width: int = 0
    height: int = 0
    faces: list[FaceBox] = field(default_factory=list)
    people: list[PersonState] = field(default_factory=list)
    signals: list[FatigueSignals] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def worst_level(self) -> FatigueLevel:
        if not self.people:
            return FatigueLevel.UNKNOWN
        return max((p.level for p in self.people), key=lambda l: l.severity)

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "latency_ms": round(self.latency_ms, 2),
            "worst_level": self.worst_level.value,
            "faces": [f.to_dict() for f in self.faces],
            "people": [p.to_dict() for p in self.people],
            "signals": [s.to_dict() for s in self.signals],
        }
