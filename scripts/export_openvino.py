"""Export weights YOLOv8 (.pt) ke OpenVINO IR untuk inference cepat di Intel.

Menghasilkan dua varian:

    models/best_openvino_model/       FP32 — akurasi identik dengan .pt
    models/best_int8_openvino_model/  INT8 — hasil post-training quantization
                                      (NNCF), ~2-4x lebih cepat, ukuran ~4x
                                      lebih kecil, dengan sedikit penurunan mAP

Quantization INT8 butuh data kalibrasi: NNCF menjalankan beberapa ratus gambar
lewat model untuk mengukur rentang aktivasi tiap layer. Ultralytics mengambilnya
dari split `val` pada data.yaml, jadi dataset harus sudah ter-download.

Contoh:
    python scripts/export_openvino.py                    # FP32 + INT8
    python scripts/export_openvino.py --no-int8          # FP32 saja (tanpa dataset)
    python scripts/export_openvino.py --imgsz 416
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_WEIGHTS = PROJECT_ROOT / "models" / "best.pt"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "ppe-detection-2" / "data.yaml"


def resolve_imgsz(model, override: int | None) -> int:
    """Pakai imgsz yang sama dengan saat training kalau tidak di-override.

    Export pada resolusi berbeda dari training menurunkan akurasi tanpa
    peringatan apa pun. Ultralytics menyimpan konfigurasi training di dalam
    checkpoint (`train_args`), jadi nilainya dibaca dari situ — lebih andal
    daripada menebak dari isi folder runs/.
    """
    if override:
        return override
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args") or {}
    imgsz = train_args.get("imgsz")
    if imgsz:
        print(f"[INFO] imgsz={imgsz} diambil dari metadata checkpoint.")
        return int(imgsz)
    print("[WARN] imgsz training tidak terdeteksi di checkpoint, fallback ke 640.")
    return 640


def int8_kwargs() -> dict:
    """Argumen INT8 yang sesuai versi Ultralytics yang terpasang.

    Ultralytics 8.4 mengganti flag boolean `int8=True` dengan `quantize=8`
    (yang juga menerima 16, 'w8a16', dst). Versi lama hanya kenal `int8`, dan
    versi baru masih menerimanya tapi memuntahkan DeprecationWarning tiap
    export. Dipilih di runtime supaya requirements tetap `ultralytics>=8.2.0`.
    """
    try:
        from ultralytics.cfg import DEFAULT_CFG_DICT

        if "quantize" in DEFAULT_CFG_DICT:
            return {"quantize": 8}
    except ImportError:
        pass
    return {"int8": True}


def export(model, weights: Path, imgsz: int, int8: bool, data: Path | None) -> Path:
    kind = "INT8" if int8 else "FP32"
    print(f"[INFO] Export {kind} | weights={weights} | imgsz={imgsz}")

    kwargs: dict = {"format": "openvino", "imgsz": imgsz}
    if int8:
        if data is None or not data.exists():
            raise SystemExit(
                f"[ERROR] Quantization INT8 butuh dataset kalibrasi, tapi {data} "
                "tidak ada. Jalankan `python scripts/download_dataset.py`, atau "
                "pakai --no-int8."
            )
        kwargs.update(int8_kwargs())
        kwargs["data"] = str(data)

    out = Path(model.export(**kwargs))
    print(f"[OK] {kind} tersimpan di {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Export YOLOv8 ke OpenVINO IR")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Default: dibaca dari args.yaml run training")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA,
                    help="data.yaml untuk kalibrasi INT8")
    ap.add_argument("--no-int8", dest="int8", action="store_false",
                    help="Lewati varian INT8 (tidak butuh dataset)")
    ap.add_argument("--no-fp32", dest="fp32", action="store_false",
                    help="Lewati varian FP32")
    args = ap.parse_args()

    if not args.weights.exists():
        print(f"[ERROR] Weights tidak ditemukan: {args.weights}\n"
              f"        Latih dulu lewat `python scripts/train.py`.", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    imgsz = resolve_imgsz(YOLO(str(args.weights)), args.imgsz)
    models_dir = PROJECT_ROOT / "models"

    # Model di-load ulang per varian: `export()` mengubah state model in-place
    # (fuse layer, ganti mode), jadi memakai instance yang sama untuk FP32 lalu
    # INT8 bisa menghasilkan IR yang tidak konsisten.
    if args.fp32:
        out = export(YOLO(str(args.weights)), args.weights, imgsz, int8=False, data=None)
        _rename(out, models_dir / "best_openvino_model")
    if args.int8:
        out = export(YOLO(str(args.weights)), args.weights, imgsz, int8=True, data=args.data)
        _rename(out, models_dir / "best_int8_openvino_model")

    print("\n[NEXT] Coba jalankan:")
    print("  python -m src.cli --backend openvino image frame.jpg")
    print("  python scripts/benchmark.py")
    return 0


def _rename(src: Path, dst: Path) -> None:
    """Ultralytics menamai folder mengikuti nama weights; samakan ke nama baku."""
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.move(str(src), str(dst))
    print(f"[OK] Dipindahkan ke {dst}")


if __name__ == "__main__":
    sys.exit(main())
