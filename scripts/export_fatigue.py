"""Export classifier fatigue (.pt) ke ONNX + OpenVINO IR untuk deploy CPU.

Menghasilkan:

    models/fatigue/fatigue_cls.onnx                     perantara (opset 17)
    models/fatigue/fatigue_cls_openvino_model/          FP32 — akurasi identik
    models/fatigue/fatigue_cls_int8_openvino_model/     INT8 — ~2-4x lebih cepat

Contoh:
    python scripts/export_fatigue.py                 # ONNX + FP32 + INT8
    python scripts/export_fatigue.py --no-int8       # tanpa kuantisasi

Kuantisasi INT8 memakai NNCF dengan data kalibrasi dari split TRAIN dataset
fatigue. Sengaja train, bukan val: kalibrasi cuma mengukur rentang aktivasi
tiap layer, dan memakai val untuk itu berarti data validasi ikut membentuk
model yang lalu diukur dengan data validasi yang sama.

Ambang keputusan dan metadata arsitektur ikut ditulis ke `metadata.json` di
sebelah IR-nya. Tanpa itu, IR yang dideploy tidak tahu ambangnya sendiri dan
akan diam-diam jatuh ke 0.5 — yang bukan angka hasil tuning mana pun.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.classifier import (  # noqa: E402
    CLASSES,
    MEAN,
    STD,
    build_backbone,
    preprocess_bgr,
)

MODEL_DIR = PROJECT_ROOT / "models" / "fatigue"
DEFAULT_CKPT = MODEL_DIR / "fatigue_cls.pt"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "fatigue"
# Jumlah gambar kalibrasi. NNCF butuh cukup sampel untuk mengukur rentang
# aktivasi; 300 sudah jauh di atas titik jenuhnya untuk model sekecil ini.
CALIBRATION_SAMPLES = 300


def load_checkpoint(path: Path):
    import torch

    if not path.exists():
        raise SystemExit(
            f"[ERROR] Checkpoint tidak ada: {path}\n"
            "        Jalankan dulu: python scripts/train_fatigue.py"
        )
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_backbone(ckpt["arch"], num_classes=len(CLASSES), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def export_onnx(model, out_path: Path, image_size: int, opset: int) -> Path:
    """Export ke ONNX lewat exporter TorchScript (dynamo=False).

    torch >= 2.9 memakai exporter berbasis `torch.export` (dynamo) secara
    default. Untuk model ini exporter itu bermasalah dua kali: ia mengabaikan
    `dynamic_axes` (memintanya diganti `dynamic_shapes` dengan sintaks
    berbeda), dan logger-nya mencetak emoji ke stdout — yang di konsol Windows
    ber-codepage cp1252 melempar UnicodeEncodeError dan menggagalkan export
    yang sebenarnya sudah berhasil.

    Exporter lama tidak punya keduanya dan sudah lebih dari cukup untuk CNN
    klasifikasi biasa tanpa kontrol aliran dinamis.
    """
    import torch

    out_path.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, image_size, image_size)
    torch.onnx.export(
        model, dummy, str(out_path),
        input_names=["images"], output_names=["logits"],
        opset_version=opset,
        dynamo=False,
        # Batch dinamis: pipeline memberi 1..8 wajah per frame, dan IR dengan
        # batch tetap akan menolak semua kecuali satu ukuran itu.
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    )
    print(f"[OK] ONNX: {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    return out_path


def calibration_images(data_dir: Path, image_size: int, limit: int) -> list[np.ndarray]:
    """Ambil gambar kalibrasi dari split train, seimbang antar kelas."""
    import cv2

    train = data_dir / "train"
    if not train.is_dir():
        raise SystemExit(
            f"[ERROR] {train} tidak ada — kalibrasi INT8 butuh dataset.\n"
            "        Jalankan scripts/prepare_fatigue_dataset.py, "
            "atau export dengan --no-int8."
        )
    per_class = max(1, limit // len(CLASSES))
    batch = []
    for label in CLASSES:
        files = sorted((train / label).glob("*.jpg"))[:per_class]
        for path in files:
            img = cv2.imread(str(path))
            if img is not None:
                batch.append(preprocess_bgr(img, image_size)[None, ...])
    if not batch:
        raise SystemExit(f"[ERROR] Tidak ada gambar kalibrasi di {train}")
    return batch


def write_metadata(directory: Path, ckpt: dict, image_size: int, precision: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps({
        "arch": ckpt.get("arch"),
        "classes": list(CLASSES),
        "image_size": image_size,
        "threshold": ckpt.get("threshold"),
        "precision": precision,
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        "source_checkpoint": str(DEFAULT_CKPT.name),
        "val_metrics": ckpt.get("val_metrics"),
        "test_metrics": ckpt.get("test_metrics"),
        "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out-dir", type=Path, default=MODEL_DIR)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-int8", action="store_true", help="Lewati kuantisasi INT8.")
    ap.add_argument("--samples", type=int, default=CALIBRATION_SAMPLES)
    args = ap.parse_args()

    import openvino as ov

    model, ckpt = load_checkpoint(args.checkpoint)
    image_size = int(ckpt.get("image_size", 224))
    print(f"[INFO] {ckpt.get('arch')} — input {image_size}px, "
          f"ambang {ckpt.get('threshold', 0.5):.4f}")

    onnx_path = export_onnx(model, args.out_dir / "fatigue_cls.onnx", image_size, args.opset)

    # ---- FP32 ----
    ov_model = ov.convert_model(str(onnx_path))
    fp32_dir = args.out_dir / "fatigue_cls_openvino_model"
    if fp32_dir.exists():
        shutil.rmtree(fp32_dir)
    fp32_dir.mkdir(parents=True)
    ov.save_model(ov_model, str(fp32_dir / "fatigue_cls.xml"), compress_to_fp16=False)
    write_metadata(fp32_dir, ckpt, image_size, "FP32")
    size = sum(f.stat().st_size for f in fp32_dir.iterdir()) / 1e6
    print(f"[OK] OpenVINO FP32: {fp32_dir} ({size:.1f} MB)")

    # ---- INT8 ----
    if not args.no_int8:
        import nncf

        batches = calibration_images(args.data, image_size, args.samples)
        print(f"[INFO] Kuantisasi INT8 dengan {len(batches)} gambar kalibrasi…")
        quantized = nncf.quantize(
            ov_model,
            nncf.Dataset(batches),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(batches),
        )
        int8_dir = args.out_dir / "fatigue_cls_int8_openvino_model"
        if int8_dir.exists():
            shutil.rmtree(int8_dir)
        int8_dir.mkdir(parents=True)
        ov.save_model(quantized, str(int8_dir / "fatigue_cls.xml"))
        write_metadata(int8_dir, ckpt, image_size, "INT8")
        size = sum(f.stat().st_size for f in int8_dir.iterdir()) / 1e6
        print(f"[OK] OpenVINO INT8: {int8_dir} ({size:.1f} MB)")

    print("\n[INFO] Pakai dengan:\n"
          "    FATIGUE_BACKEND=openvino-int8 python -m src.fatigue.cli webcam\n"
          "atau set FATIGUE_BACKEND di .env")
    return 0


if __name__ == "__main__":
    sys.exit(main())
