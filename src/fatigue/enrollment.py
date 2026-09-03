"""Validasi & pendaftaran foto wajah — satu definisi untuk CLI, UI, dan API.

Ketiga jalur pendaftaran (script CLI, halaman Streamlit, endpoint FastAPI)
sebelumnya punya aturannya masing-masing, dan ketiganya berbeda: yang satu
mengecek ukuran wajah minimum, yang lain tidak; yang satu menolak foto kembar,
yang lain menyimpannya. Akibatnya karyawan yang didaftarkan lewat UI mendapat
kualitas pendaftaran yang lebih rendah daripada yang lewat CLI, tanpa ada yang
memberitahunya.

Itu masalah nyata, karena **kualitas pendaftaran adalah pengungkit terbesar
pada keandalan absensi** — lebih besar daripada pilihan model. Jadi aturannya
dikumpulkan di sini dan ketiga jalur memanggil yang sama.

Prinsip yang dipakai: tolak hanya yang terbukti membuat foto tidak terpakai,
peringatkan sisanya. Orang yang gagal mendaftar sama sekali tidak akan pernah
dikenali kamera, jadi gerbang yang terlalu ketat lebih merugikan daripada foto
yang kurang ideal.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.fatigue.face import FaceDetector, FaceEmbedder
from src.fatigue.types import FaceBox

# Wajah di bawah ini terlalu sedikit pikselnya untuk embedding yang stabil.
# Lebih besar dari ambang deteksi runtime (40 px) dengan sengaja: foto
# pendaftaran bisa dipilih, jadi tidak ada alasan menerima yang seadanya.
MIN_ENROLL_FACE = 80

# Kemiripan maksimum antar foto pendaftaran orang yang sama. Di atas ini foto
# barunya nyaris duplikat — ia tidak menambah cakupan pose apa pun, cuma
# memperbesar database dan memperlambat pencocokan.
DUPLICATE_SIM = 0.97

# Ketajaman di bawah ini memicu PERINGATAN, bukan penolakan.
#
# Ambang ini semula 40 dan MENOLAK foto. Itu keliru, dan terukur keliru:
# sebuah frame webcam 640x480 dengan wajah frontal dan jelas — yang terbukti
# menghasilkan pengenalan sempurna — hanya mendapat nilai 17, sementara foto
# internet terkurasi mendapat 90-1600. Metrik ini lebih banyak mengukur ASAL
# gambar (sensor webcam yang lembut vs foto yang sudah dipertajam) daripada
# kelayakannya untuk dikenali.
#
# Yang menentukan keputusan: embedding wajah ternyata sangat tahan blur. Diuji
# dengan blur Gaussian bertingkat pada foto yang sama, similarity terhadap
# versi aslinya masih 0,62 pada kernel 21x21 — jauh di atas ambang pengenalan
# 0,40 — padahal ketajamannya sudah runtuh. Rentang yang berguna terlalu sempit
# untuk dijadikan gerbang penolakan.
SOFT_SHARPNESS = 8.0

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class PhotoResult:
    """Hasil penilaian satu foto pendaftaran."""

    name: str
    accepted: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    face: FaceBox | None = None
    sharpness: float | None = None

    def describe(self) -> str:
        if not self.accepted:
            return f"ditolak — {self.reason}"
        return "; ".join(self.warnings) if self.warnings else "ok"


@dataclass
class EnrollResult:
    """Hasil pendaftaran satu karyawan."""

    employee_id: str
    name: str
    photos: list[PhotoResult] = field(default_factory=list)

    @property
    def accepted(self) -> int:
        return sum(1 for p in self.photos if p.accepted)

    @property
    def rejected(self) -> list[PhotoResult]:
        return [p for p in self.photos if not p.accepted]

    @property
    def warned(self) -> list[PhotoResult]:
        return [p for p in self.photos if p.accepted and p.warnings]

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "accepted": self.accepted,
            "rejected": [{"file": p.name, "reason": p.reason} for p in self.rejected],
            "warnings": [{"file": p.name, "warnings": p.warnings} for p in self.warned],
        }


def face_sharpness(img: np.ndarray, face: FaceBox, size: int = 160) -> float:
    """Variance Laplacian pada crop wajah yang ukurannya sudah dinormalisasi.

    Crop diskalakan ke ukuran tetap dulu supaya angkanya sebanding antar
    resolusi. Tanpa itu, wajah kecil selalu terlihat lebih buram daripada wajah
    besar yang sama tajamnya, dan ambang apa pun akan bergantung pada jarak
    orang ke kamera alih-alih pada kualitas fotonya.

    Diukur pada area wajah saja: latar yang tajam bisa menutupi wajah yang
    buram kalau dihitung se-frame.
    """
    x1, y1, x2, y2 = face.bbox
    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def check_photo(img: np.ndarray, faces: list[FaceBox], name: str = "") -> PhotoResult:
    """Nilai satu foto pendaftaran terhadap aturan bersama."""
    if img is None or not getattr(img, "size", 0):
        return PhotoResult(name=name, accepted=False, reason="file tidak terbaca")
    if not faces:
        return PhotoResult(name=name, accepted=False, reason="wajah tidak terdeteksi")
    if len(faces) > 1:
        return PhotoResult(
            name=name, accepted=False,
            reason=f"ada {len(faces)} wajah — foto pendaftaran harus satu orang",
        )

    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    width, height = x2 - x1, y2 - y1
    if min(width, height) < MIN_ENROLL_FACE:
        return PhotoResult(
            name=name, accepted=False, face=face,
            reason=f"wajah terlalu kecil ({width}x{height} px, "
                   f"minimal {MIN_ENROLL_FACE})",
        )

    sharpness = face_sharpness(img, face)
    warnings = []
    if sharpness < SOFT_SHARPNESS:
        warnings.append(
            f"agak buram (ketajaman {sharpness:.1f}) — tetap dipakai, tapi foto "
            "yang lebih tajam akan lebih andal"
        )
    return PhotoResult(name=name, accepted=True, face=face,
                       sharpness=sharpness, warnings=warnings)


def enroll_photos(
    book,
    detector: FaceDetector,
    embedder: FaceEmbedder,
    employee_id: str,
    name: str,
    department: str,
    photos: list[tuple[str, np.ndarray]],
) -> EnrollResult:
    """Daftarkan sekumpulan (nama_file, gambar BGR) untuk satu karyawan.

    Karyawannya dibuat lebih dulu meski semua fotonya ditolak — supaya ia
    muncul di daftar dengan "0 foto" dan kegagalannya terlihat, alih-alih
    menghilang tanpa jejak dan ditemukan berminggu-minggu kemudian sebagai
    "kok orang ini tidak pernah terbaca".
    """
    book.add_employee(employee_id, name, department)
    result = EnrollResult(employee_id=employee_id, name=name)
    accepted_vectors: list[np.ndarray] = []

    for filename, img in photos:
        faces = detector.detect(img) if img is not None else []
        check = check_photo(img, faces, name=filename)
        if not check.accepted:
            result.photos.append(check)
            continue

        vector = embedder.embed(img, faces[0])
        duplicate = next(
            (i for i, v in enumerate(accepted_vectors)
             if float(np.dot(v, vector)) >= DUPLICATE_SIM),
            None,
        )
        if duplicate is not None:
            result.photos.append(PhotoResult(
                name=filename, accepted=False, face=faces[0],
                reason=f"nyaris identik dengan foto ke-{duplicate + 1} — "
                       "ubah sudut atau pencahayaan",
            ))
            continue

        book.add_embedding(employee_id, vector, source=filename)
        accepted_vectors.append(vector)
        result.photos.append(check)

    return result
