"""Training YOLOv8 lokal di atas dataset PPE hasil export Roboflow.

Alur:
    1. `python scripts/download_dataset.py`  -> datasets/ppe-detection-2/
    2. `python scripts/train.py --epochs 30` -> runs/ppe/<name>/weights/best.pt
    3. Weights terbaik otomatis disalin ke models/best.pt

Contoh:
    # full dataset, butuh GPU
    python scripts/train.py --epochs 50 --imgsz 640 --batch 16

    # smoke test cepat di CPU (subset kecil, resolusi kecil)
    python scripts/train.py --epochs 3 --imgsz 320 --batch 8 --subset 300
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "ppe-detection-2"


def build_subset(data_yaml: Path, n_train: int, n_val: int, out_dir: Path) -> Path:
    """Bikin dataset kecil (symlink/copy) untuk smoke test di CPU.

    Mengembalikan path data.yaml baru yang menunjuk ke subset.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent
    rng = random.Random(42)

    if out_dir.exists():
        shutil.rmtree(out_dir)

    for split, limit in (("train", n_train), ("valid", n_val)):
        src_img = root / split / "images"
        src_lbl = root / split / "labels"
        if not src_img.exists():
            continue
        images = sorted(src_img.glob("*.jpg")) + sorted(src_img.glob("*.png"))
        rng.shuffle(images)
        picked = images[:limit]

        dst_img = out_dir / split / "images"
        dst_lbl = out_dir / split / "labels"
        dst_img.mkdir(parents=True, exist_ok=True)
        dst_lbl.mkdir(parents=True, exist_ok=True)
        for img in picked:
            shutil.copy2(img, dst_img / img.name)
            lbl = src_lbl / f"{img.stem}.txt"
            if lbl.exists():
                shutil.copy2(lbl, dst_lbl / lbl.name)
        print(f"[subset] {split}: {len(picked)} gambar")

    new_cfg = {
        "names": cfg["names"],
        "nc": cfg["nc"],
        "train": str((out_dir / "train" / "images").resolve()),
        "val": str((out_dir / "valid" / "images").resolve()),
    }
    out_yaml = out_dir / "data.yaml"
    out_yaml.write_text(yaml.safe_dump(new_cfg, sort_keys=False), encoding="utf-8")
    return out_yaml


def publish_best(name: str) -> int:
    """Salin best.pt hasil run ke models/best.pt."""
    best = PROJECT_ROOT / "runs" / "ppe" / name / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt tidak ditemukan di {best}", file=sys.stderr)
        return 1
    target = PROJECT_ROOT / "models" / "best.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print(f"[OK] Weights terbaik disalin ke {target}")
    return 0


def resume(name: str) -> int:
    """Lanjutkan run yang terputus dari checkpoint `last.pt`.

    Ultralytics menyimpan seluruh konfigurasi (dataset, epochs, imgsz, optimizer
    state) di dalam checkpoint, jadi tidak ada argumen train lain yang perlu
    diulang di sini — kalau di-override malah dianggap konflik.
    """
    last = PROJECT_ROOT / "runs" / "ppe" / name / "weights" / "last.pt"
    if not last.exists():
        print(f"[ERROR] Checkpoint tidak ditemukan di {last}", file=sys.stderr)
        return 1

    from ultralytics import YOLO

    print(f"[INFO] Melanjutkan training dari {last}")
    YOLO(str(last)).train(resume=True)
    return publish_best(name)


def main() -> int:
    ap = argparse.ArgumentParser(description="Train YOLOv8 untuk PPE Detection")
    ap.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                    help="Folder dataset hasil export Roboflow (berisi data.yaml)")
    ap.add_argument("--model", type=str, default="yolov8n.pt",
                    help="Weights awal (yolov8n/s/m.pt) atau .yaml untuk from scratch")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", type=str, default=None,
                    help="cuda / cpu / 0. Default: auto-detect")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", type=str, default="ppe-yolov8n")
    ap.add_argument("--subset", type=int, default=0,
                    help="Kalau >0, latih hanya pada N gambar train (smoke test CPU)")
    ap.add_argument("--subset-val", type=int, default=100)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true",
                    help="Lanjutkan run yang terputus dari runs/ppe/<name>/weights/last.pt")
    args = ap.parse_args()

    if args.resume:
        return resume(args.name)

    data_yaml = args.dataset / "data.yaml"
    if not data_yaml.exists():
        print(f"[ERROR] data.yaml tidak ditemukan di {data_yaml}.\n"
              f"        Jalankan `python scripts/download_dataset.py` dulu.", file=sys.stderr)
        return 1

    if args.subset:
        data_yaml = build_subset(
            data_yaml, args.subset, args.subset_val,
            PROJECT_ROOT / "datasets" / f"_subset_{args.subset}",
        )

    import torch
    from ultralytics import YOLO

    device = args.device or ("0" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device} | dataset={data_yaml} | epochs={args.epochs} "
          f"| imgsz={args.imgsz} | batch={args.batch}")
    if device == "cpu":
        print("[WARN] Training di CPU jauh lebih lambat. Untuk dataset penuh "
              "(~20k gambar) gunakan GPU (Colab/RunPod) atau pakai --subset.")

    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        project=str(PROJECT_ROOT / "runs" / "ppe"),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        plots=True,
    )

    return publish_best(args.name)


if __name__ == "__main__":
    sys.exit(main())
