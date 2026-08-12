"""Evaluasi akurasi model pada split test — termasuk efek quantization INT8.

Tujuannya menjawab satu pertanyaan praktis: OpenVINO INT8 memang lebih cepat,
tapi berapa mAP yang hilang? Angka dari sini dipasangkan dengan hasil
`scripts/benchmark.py` untuk memilih backend yang dipakai di lapangan.

Contoh:
    python scripts/evaluate.py
    python scripts/evaluate.py --split val --variants pt,openvino
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATA = PROJECT_ROOT / "datasets" / "ppe-detection-2" / "data.yaml"

VARIANTS = {
    "pt": ("PyTorch FP32", PROJECT_ROOT / "models" / "best.pt"),
    "openvino": ("OpenVINO FP32", PROJECT_ROOT / "models" / "best_openvino_model"),
    "openvino-int8": ("OpenVINO INT8", PROJECT_ROOT / "models" / "best_int8_openvino_model"),
}


def build_eval_yaml(data_yaml: Path, split: str) -> Path:
    """Bikin data.yaml sementara yang `val`-nya menunjuk ke split yang diminta.

    Ultralytics selalu mengevaluasi key `val`, jadi untuk menilai split `test`
    kita perlu menuliskannya ulang alih-alih mengandalkan argumen split.
    """
    cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = data_yaml.parent

    target = root / split / "images"
    if not target.exists():
        raise SystemExit(f"[ERROR] Split '{split}' tidak ada di {root}")

    out = PROJECT_ROOT / "datasets" / f"_eval_{split}.yaml"
    out.write_text(
        yaml.safe_dump(
            {"names": cfg["names"], "nc": cfg["nc"],
             "train": str(target.resolve()), "val": str(target.resolve())},
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return out


def evaluate(model_path: Path, data_yaml: Path, imgsz: int, batch: int) -> dict | None:
    from ultralytics import YOLO

    if not model_path.exists():
        print(f"[SKIP] {model_path} belum ada.")
        return None

    print(f"[RUN ] Evaluasi {model_path.name} ...", flush=True)
    model = YOLO(str(model_path), task="detect")
    m = model.val(data=str(data_yaml), imgsz=imgsz, batch=batch,
                  verbose=False, plots=False).box
    return {
        "mAP50": float(m.map50),
        "mAP50-95": float(m.map),
        "precision": float(m.mp),
        "recall": float(m.mr),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluasi akurasi tiap varian model")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", type=str, default="test", choices=["test", "valid", "train"])
    ap.add_argument("--imgsz", type=int, default=None,
                    help="Default: dibaca dari checkpoint models/best.pt")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--variants", type=str, default="pt,openvino,openvino-int8")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "METRICS.md")
    args = ap.parse_args()

    if not args.data.exists():
        print(f"[ERROR] {args.data} tidak ada. Jalankan scripts/download_dataset.py.",
              file=sys.stderr)
        return 1

    imgsz = args.imgsz
    if imgsz is None:
        from ultralytics import YOLO

        pt = VARIANTS["pt"][1]
        if not pt.exists():
            print(f"[ERROR] {pt} tidak ada dan --imgsz tidak diberikan.", file=sys.stderr)
            return 1
        imgsz = int((getattr(YOLO(str(pt)), "ckpt", None) or {})
                    .get("train_args", {}).get("imgsz", 640))
        print(f"[INFO] imgsz={imgsz} dari checkpoint.")

    eval_yaml = build_eval_yaml(args.data, args.split)
    wanted = [v.strip() for v in args.variants.split(",") if v.strip()]

    rows = []
    for key in wanted:
        if key not in VARIANTS:
            print(f"[SKIP] Varian '{key}' tidak dikenal.")
            continue
        label, path = VARIANTS[key]
        metrics = evaluate(path, eval_yaml, imgsz, args.batch)
        if metrics:
            rows.append({"label": label, **metrics})
            print(f"[DONE] {label}: mAP50={metrics['mAP50']:.4f} "
                  f"mAP50-95={metrics['mAP50-95']:.4f}\n")

    if not rows:
        print("[ERROR] Tidak ada model yang bisa dievaluasi.", file=sys.stderr)
        return 1

    base = rows[0]["mAP50"]
    lines = [
        "# Metrics Akurasi — PPE Detection",
        "",
        f"Split: **{args.split}** · imgsz **{imgsz}** · dataset `{args.data.parent.name}`",
        "",
        "| Varian | mAP@50 | mAP@50-95 | Precision | Recall | Δ mAP@50 |",
        "|--------|--------|-----------|-----------|--------|----------|",
    ]
    for r in rows:
        delta = "—" if r["mAP50"] == base else f"{(r['mAP50'] - base) * 100:+.2f} pp"
        lines.append(
            f"| {r['label']} | {r['mAP50']:.4f} | {r['mAP50-95']:.4f} | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {delta} |"
        )
    lines += ["", "> Δ dihitung terhadap varian pertama (PyTorch FP32).", ""]

    md = "\n".join(lines)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[OK] Ditulis ke {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
