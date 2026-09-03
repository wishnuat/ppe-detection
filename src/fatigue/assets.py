"""Unduh & verifikasi bobot pihak ketiga yang dipakai pipeline fatigue.

Kenapa tidak di-commit seperti `models/best.pt`: SFace saja 37 MB dan ketiganya
punya lisensi + sumber resmi sendiri, jadi lebih jujur (dan lebih ramah ukuran
repo) untuk mengambilnya dari upstream. Yang di-commit hanya *pointer*-nya:
URL + sha256 di tabel bawah, sehingga hasil unduhan bisa diverifikasi dan
build tetap reproducible.

Pakai:
    python -m src.fatigue.assets            # unduh semua yang belum ada
    python -m src.fatigue.assets --verify   # cek ulang sha256 yang sudah ada

Di kode, panggil `ensure("yunet")` — file dikembalikan kalau sudah ada,
diunduh sekali kalau belum.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Bisa dipindah ke volume/cache bersama di Docker lewat env var.
ASSET_DIR = Path(os.getenv("FATIGUE_ASSET_DIR", PROJECT_ROOT / "models" / "fatigue"))


@dataclass(frozen=True)
class Asset:
    name: str
    filename: str
    url: str
    sha256: str
    size: int
    license: str
    note: str


ASSETS: dict[str, Asset] = {
    "yunet": Asset(
        name="yunet",
        filename="face_detection_yunet_2023mar.onnx",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        size=232589,
        license="MIT (OpenCV Zoo)",
        note="Detektor wajah 227 KB, dijalankan lewat cv2.FaceDetectorYN.",
    ),
    "sface": Asset(
        name="sface",
        filename="face_recognition_sface_2021dec.onnx",
        url="https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        size=38696353,
        license="Apache-2.0 (OpenCV Zoo)",
        note="Embedding wajah 128-d, cv2.FaceRecognizerSF. ~99.6% LFW.",
    ),
    "face_landmarker": Asset(
        name="face_landmarker",
        filename="face_landmarker.task",
        url="https://storage.googleapis.com/mediapipe-models/face_landmarker/"
            "face_landmarker/float16/1/face_landmarker.task",
        sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
        size=3758596,
        license="Apache-2.0 (Google MediaPipe)",
        note="478 landmark + 52 blendshape + matriks transformasi kepala.",
    ),
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def path_for(name: str) -> Path:
    return ASSET_DIR / ASSETS[name].filename


def is_valid(name: str) -> bool:
    """True kalau file ada, ukurannya pas, dan sha256-nya cocok."""
    asset = ASSETS[name]
    p = path_for(name)
    if not p.exists() or p.stat().st_size != asset.size:
        return False
    return _sha256(p) == asset.sha256


def ensure(name: str, *, quiet: bool = False) -> Path:
    """Kembalikan path bobot `name`, unduh dulu kalau belum ada/rusak.

    Unduhan ditulis ke file `.part` lalu di-rename — supaya Ctrl-C atau koneksi
    putus tidak meninggalkan file setengah jadi yang lolos cek `exists()` di
    proses berikutnya.
    """
    if name not in ASSETS:
        raise KeyError(f"Asset '{name}' tidak dikenal. Pilihan: {', '.join(ASSETS)}")
    asset = ASSETS[name]
    dest = path_for(name)

    if dest.exists() and is_valid(name):
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if not quiet:
        mb = asset.size / 1e6
        print(f"[INFO] Mengunduh {asset.filename} ({mb:.1f} MB) — {asset.license}")
    try:
        with urllib.request.urlopen(asset.url, timeout=120) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Gagal mengunduh {asset.filename} dari {asset.url}: {exc}\n"
            f"Unduh manual lalu simpan ke {dest}."
        ) from exc

    got = _sha256(tmp)
    if got != asset.sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"sha256 {asset.filename} tidak cocok (dapat {got[:16]}…, "
            f"harusnya {asset.sha256[:16]}…). File di server berubah atau unduhan rusak."
        )
    tmp.replace(dest)
    if not quiet:
        print(f"[OK] Tersimpan: {dest}")
    return dest


def ensure_all(quiet: bool = False) -> dict[str, Path]:
    return {name: ensure(name, quiet=quiet) for name in ASSETS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--verify", action="store_true",
                    help="Hanya cek sha256 file yang sudah ada, tanpa mengunduh.")
    ap.add_argument("--only", type=str, default=None,
                    help=f"Batasi ke satu asset: {', '.join(ASSETS)}")
    args = ap.parse_args()

    names = [args.only] if args.only else list(ASSETS)
    rc = 0
    for name in names:
        asset = ASSETS[name]
        if args.verify:
            p = path_for(name)
            if not p.exists():
                print(f"[MISS] {asset.filename} belum diunduh")
                rc = 1
            elif is_valid(name):
                print(f"[OK]   {asset.filename}")
            else:
                print(f"[BAD]  {asset.filename} — sha256/ukuran tidak cocok")
                rc = 1
        else:
            ensure(name)
    return rc


if __name__ == "__main__":
    sys.exit(main())
