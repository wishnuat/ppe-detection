"""Pilih ulang ambang keputusan classifier fatigue, tanpa melatih ulang.

    python scripts/tune_fatigue_threshold.py                    # lihat pilihan
    python scripts/tune_fatigue_threshold.py --apply            # tulis ke checkpoint
    python scripts/tune_fatigue_threshold.py --criterion recall-floor --recall-floor 0.95

Ambang keputusan adalah **kebijakan operasional, bukan properti model**. Model
yang sama, ambang berbeda, memberi keputusan yang sangat berbeda — dan
pertukaran yang tepat antara "melewatkan orang yang mengantuk" dan
"membunyikan alarm palsu" tidak sama untuk operator crane, sopir truk, dan
petugas gudang. Karena itu ia diberi entry point sendiri: bisa diputar ulang
kapan saja, oleh orang yang paham konteks lokasinya, tanpa menyentuh bobot.

PENTING — apa yang ambang ini pengaruhi, dan apa yang tidak
-----------------------------------------------------------
`FatiguePipeline` **tidak memakai ambang ini.** Fusi mengonsumsi probabilitas
mentah classifier sebagai bukti kontinu, bukan keputusan biner. Titik operasi
sistem yang sebenarnya diatur `FusionConfig.mild_at / severe_at / critical_at`
(bisa diubah dari sidebar Streamlit tanpa restart).

Yang dipengaruhi ambang ini: angka di `scripts/evaluate_fatigue.py`, laporan
akurasi, dan siapa pun yang memakai `FatigueClassifier` secara terpisah dari
pipeline. Itu tetap berguna — akurasi yang dilaporkan harus mencerminkan
operating point yang masuk akal — tapi jangan berharap mengubahnya mengubah
perilaku alarm di lapangan.

Ambang dipilih di **validation set**. Test set hanya dilaporkan sesudahnya,
sebagai perkiraan jujur tentang apa yang akan terjadi — bukan sebagai bahan
pemilihan. Kalau ambang dipilih di test set, angka test-nya berhenti berarti.

Perubahan ditulis ke `models/fatigue/fatigue_cls.pt` dan ke `metadata.json` di
setiap folder IR OpenVINO, sehingga semua backend memakai titik operasi yang
sama. Tanpa itu, mengganti `FATIGUE_BACKEND` diam-diam mengganti kebijakan.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.classifier import (  # noqa: E402
    CLASSES,
    DEFAULT_CHECKPOINT,
    DEFAULT_FN_COST,
    DEFAULT_MODEL_DIR,
    THRESHOLD_CRITERIA,
    build_classifier,
    choose_threshold,
)

DEFAULT_DATA = PROJECT_ROOT / "datasets" / "fatigue"


def load_split(data_dir: Path, split: str) -> tuple[list[np.ndarray], np.ndarray]:
    root = data_dir / split
    if not root.is_dir():
        raise SystemExit(
            f"[ERROR] {root} tidak ada. "
            "Jalankan dulu: python scripts/prepare_fatigue_dataset.py"
        )
    images, labels = [], []
    for class_index, label in enumerate(CLASSES):
        for path in sorted((root / label).glob("*.jpg")):
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
                labels.append(class_index)
    if not images:
        raise SystemExit(f"[ERROR] Tidak ada gambar di {root}")
    return images, np.asarray(labels, dtype=np.int64)


def predict(classifier, images: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    probs = np.zeros(len(images), dtype=np.float32)
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        probs[start:start + len(chunk)] = classifier.predict_batch(chunk)
    return probs


def report(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    pred = probs >= threshold
    tp = int((pred & (targets == 1)).sum())
    fp = int((pred & (targets == 0)).sum())
    fn = int((~pred & (targets == 1)).sum())
    tn = int((~pred & (targets == 0)).sum())
    recall = tp / (tp + fn) if tp + fn else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round((tp + tn) / len(targets), 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "specificity": round(tn / (tn + fp), 4) if tn + fp else 0.0,
        "f1": round(2 * precision * recall / (precision + recall), 4)
              if precision + recall else 0.0,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def write_threshold(checkpoint: Path, threshold: float, diagnostics: dict,
                    val: dict, test: dict) -> list[Path]:
    """Tulis ambang baru ke checkpoint .pt dan ke metadata tiap IR OpenVINO."""
    import torch

    touched = []
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    ckpt["threshold"] = threshold
    ckpt["threshold_selection"] = diagnostics
    ckpt["val_metrics_at_threshold"] = val
    ckpt["test_metrics_at_threshold"] = test
    torch.save(ckpt, checkpoint)
    touched.append(checkpoint)

    for directory in sorted(DEFAULT_MODEL_DIR.glob("fatigue_cls*_openvino_model")):
        meta_file = directory / "metadata.json"
        if not meta_file.exists():
            continue
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["threshold"] = threshold
        meta["threshold_selection"] = diagnostics
        meta_file.write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                             encoding="utf-8")
        touched.append(meta_file)
    return touched


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--backend", type=str, default="torch",
                    help="Backend yang dipakai menghitung probabilitas. "
                         "Pakai 'torch' kecuali IR-nya sengaja mau di-tune sendiri.")
    ap.add_argument("--criterion", choices=THRESHOLD_CRITERIA, default="cost")
    ap.add_argument("--fn-cost", type=float, default=DEFAULT_FN_COST,
                    help="Berapa kali lebih mahal satu false negative dibanding "
                         "satu false positive. Hanya untuk --criterion cost.")
    ap.add_argument("--recall-floor", type=float, default=0.93,
                    help="Target recall minimum. Hanya untuk --criterion recall-floor.")
    ap.add_argument("--apply", action="store_true",
                    help="Tulis ambang terpilih ke checkpoint & metadata IR. "
                         "Tanpa flag ini, script hanya melaporkan.")
    args = ap.parse_args()

    classifier = build_classifier(args.backend)
    print(f"[INFO] {classifier.describe()}")
    print(f"[INFO] Ambang yang berlaku sekarang: {classifier.threshold:.4f}")

    print("[INFO] Menghitung probabilitas val & test…")
    val_images, val_targets = load_split(args.data, "val")
    test_images, test_targets = load_split(args.data, "test")
    val_probs = predict(classifier, val_images)
    test_probs = predict(classifier, test_images)

    # --- perbandingan semua kriteria, supaya pilihannya terlihat, bukan diyakini ---
    print(f"\n{'kriteria':16s} {'ambang':>8s} | {'VAL rec':>8s} {'prec':>7s} {'FN':>4s} "
          f"{'FP':>4s} | {'TEST rec':>9s} {'prec':>7s} {'acc':>7s} {'FN':>4s} {'FP':>4s}")
    print("-" * 96)
    options = {}
    for criterion in THRESHOLD_CRITERIA:
        threshold, diagnostics = choose_threshold(
            val_probs, val_targets, criterion=criterion,
            fn_cost=args.fn_cost, recall_floor=args.recall_floor,
        )
        val = report(val_probs, val_targets, threshold)
        test = report(test_probs, test_targets, threshold)
        options[criterion] = (threshold, diagnostics, val, test)
        mark = " <-" if criterion == args.criterion else ""
        print(f"{criterion:16s} {threshold:8.4f} | {val['recall']:8.4f} "
              f"{val['precision']:7.4f} {val['confusion_matrix']['fn']:4d} "
              f"{val['confusion_matrix']['fp']:4d} | {test['recall']:9.4f} "
              f"{test['precision']:7.4f} {test['accuracy']:7.4f} "
              f"{test['confusion_matrix']['fn']:4d} "
              f"{test['confusion_matrix']['fp']:4d}{mark}")

    threshold, diagnostics, val, test = options[args.criterion]
    print(f"\n[INFO] Kriteria '{args.criterion}' memilih ambang {threshold:.4f}")
    if args.criterion == "cost":
        print(f"       (satu false negative dihitung {args.fn_cost:g}x lebih mahal "
              "daripada satu false positive)")
    if diagnostics["chosen"].get("recall_floor_met") is False:
        print(f"[WARN] Tidak ada ambang yang mencapai recall {args.recall_floor:.2f} "
              f"di validation set. Recall tertinggi yang bisa dicapai: "
              f"{diagnostics['chosen']['recall']:.4f}")

    print(f"\n[TEST] acc={test['accuracy']:.4f} prec={test['precision']:.4f} "
          f"rec={test['recall']:.4f} f1={test['f1']:.4f}")
    cm = test["confusion_matrix"]
    print(f"[TEST] tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")

    if not args.apply:
        print("\n[INFO] Tidak ada yang diubah. Tambahkan --apply untuk menuliskannya.")
        return 0

    touched = write_threshold(args.checkpoint, threshold, diagnostics, val, test)
    print(f"\n[OK] Ambang {threshold:.4f} ditulis ke:")
    for path in touched:
        print(f"     {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
