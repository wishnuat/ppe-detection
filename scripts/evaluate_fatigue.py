"""Evaluasi classifier fatigue di test set, per backend.

    python scripts/evaluate_fatigue.py
    python scripts/evaluate_fatigue.py --backend torch openvino openvino-int8

Menghasilkan `outputs/fatigue/eval_<backend>.json` berisi metrik lengkap plus
sapuan ambang, dan mencetak perbandingan antar backend ke konsol.

Kenapa evaluasi terpisah dari training padahal training sudah melaporkan
metrik: yang dilaporkan training adalah metrik model PyTorch-nya. Yang
sebenarnya dideploy adalah IR OpenVINO hasil export dan kuantisasi, dan
kuantisasi INT8 MENGUBAH keluaran model. Selisihnya biasanya kecil, tapi
"biasanya kecil" bukan alasan untuk tidak mengukurnya — script ini yang
menentukan apakah INT8 layak dipakai di produksi atau tidak.

Sapuan ambang dicetak supaya keputusan operasional bisa diambil sadar: sistem
peringatan kelelahan lebih baik salah ke arah terlalu sensitif (memanggil
orang yang ternyata segar) daripada melewatkan orang yang benar-benar
mengantuk, dan titik itu tidak sama untuk setiap pabrik.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.classifier import CLASSES, build_classifier  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "datasets" / "fatigue"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "fatigue"


def load_split(data_dir: Path, split: str) -> tuple[list[np.ndarray], np.ndarray, list[str]]:
    """Muat satu split sebagai (gambar BGR, label, nama file)."""
    root = data_dir / split
    if not root.is_dir():
        raise SystemExit(
            f"[ERROR] {root} tidak ada. "
            "Jalankan dulu: python scripts/prepare_fatigue_dataset.py"
        )
    images, labels, names = [], [], []
    for class_index, label in enumerate(CLASSES):
        for path in sorted((root / label).glob("*.jpg")):
            img = cv2.imread(str(path))
            if img is None:
                continue
            images.append(img)
            labels.append(class_index)
            names.append(f"{label}/{path.name}")
    if not images:
        raise SystemExit(f"[ERROR] Tidak ada gambar di {root}")
    return images, np.asarray(labels, dtype=np.int64), names


def metrics_at(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        confusion_matrix,
        roc_auc_score,
    )

    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # Balanced accuracy: rata-rata recall kedua kelas. Dilaporkan berdampingan
    # dengan akurasi biasa karena keduanya berbeda begitu ambangnya digeser
    # jauh dari 0.5, dan akurasi biasa bisa terlihat bagus sambil melewatkan
    # hampir semua kasus lelah.
    balanced = 0.5 * (recall + specificity)
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == targets).mean()), 4),
        "balanced_accuracy": round(float(balanced), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "roc_auc": round(float(roc_auc_score(targets, probs)), 4),
        "pr_auc": round(float(average_precision_score(targets, probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def sweep(probs: np.ndarray, targets: np.ndarray) -> list[dict]:
    return [metrics_at(probs, targets, t) for t in np.arange(0.05, 1.0, 0.05)]


def curve_points(probs: np.ndarray, targets: np.ndarray) -> dict:
    """Titik ROC & PR untuk digambar di luar (docs, notebook, dashboard)."""
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(targets, probs)
    precision, recall, _ = precision_recall_curve(targets, probs)
    # Diambil sampelnya supaya JSON tidak membengkak; 200 titik sudah jauh
    # lebih rapat daripada yang bisa dibedakan mata di sebuah grafik.
    def thin(arr: np.ndarray, n: int = 200) -> list[float]:
        if len(arr) <= n:
            return [round(float(v), 5) for v in arr]
        idx = np.linspace(0, len(arr) - 1, n).astype(int)
        return [round(float(v), 5) for v in arr[idx]]

    return {
        "roc": {"fpr": thin(fpr), "tpr": thin(tpr)},
        "pr": {"precision": thin(precision), "recall": thin(recall)},
    }


def evaluate_backend(
    backend: str, images: list[np.ndarray], targets: np.ndarray,
    names: list[str], batch_size: int,
) -> dict | None:
    try:
        classifier = build_classifier(backend)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[SKIP] {backend}: {exc}")
        return None

    print(f"[INFO] {backend}: {classifier.describe()}")
    probs = np.zeros(len(images), dtype=np.float32)
    t0 = time.perf_counter()
    for start in range(0, len(images), batch_size):
        chunk = images[start:start + batch_size]
        probs[start:start + len(chunk)] = classifier.predict_batch(chunk)
    elapsed = time.perf_counter() - t0

    tuned = metrics_at(probs, targets, classifier.threshold)
    at_half = metrics_at(probs, targets, 0.5)
    worst = np.argsort(np.abs(probs - targets))[::-1][:10]

    return {
        "backend": backend,
        "model": classifier.describe(),
        "model_path": classifier.model_path,
        "threshold": classifier.threshold,
        "num_images": len(images),
        "latency_ms_per_image": round(elapsed / len(images) * 1000, 3),
        "at_tuned_threshold": tuned,
        "at_threshold_0.5": at_half,
        "threshold_sweep": sweep(probs, targets),
        "curves": curve_points(probs, targets),
        # Contoh kesalahan paling percaya diri — bahan paling berguna untuk
        # memutuskan data tambahan apa yang perlu dikumpulkan berikutnya.
        "worst_errors": [
            {"file": names[i], "true": CLASSES[targets[i]],
             "prob_fatigue": round(float(probs[i]), 4)}
            for i in worst
        ],
        "_probs": probs,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--split", type=str, default="test", choices=("train", "val", "test"))
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--backend", nargs="+",
                    default=["torch", "openvino", "openvino-int8"])
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    print(f"[INFO] Memuat split '{args.split}' dari {args.data}…")
    images, targets, names = load_split(args.data, args.split)
    print(f"[INFO] {len(images)} gambar "
          f"({int((targets == 0).sum())} nonfatigue, {int((targets == 1).sum())} fatigue)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for backend in args.backend:
        result = evaluate_backend(backend, images, targets, names, args.batch_size)
        if result is None:
            continue
        probs = result.pop("_probs")
        np.savez(args.out_dir / f"eval_probs_{backend.replace('-', '_')}.npz",
                 probs=probs, targets=targets)
        path = args.out_dir / f"eval_{backend.replace('-', '_')}.json"
        path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[OK] {path}")
        results.append(result)

    if not results:
        raise SystemExit("[ERROR] Tidak ada backend yang bisa dievaluasi.")

    print(f"\n{'backend':16s} {'acc':>7s} {'bal.acc':>8s} {'prec':>7s} "
          f"{'rec':>7s} {'f1':>7s} {'AUC':>7s} {'ms/img':>8s}")
    print("-" * 70)
    for r in results:
        m = r["at_tuned_threshold"]
        print(f"{r['backend']:16s} {m['accuracy']:7.4f} {m['balanced_accuracy']:8.4f} "
              f"{m['precision']:7.4f} {m['recall']:7.4f} {m['f1']:7.4f} "
              f"{m['roc_auc']:7.4f} {r['latency_ms_per_image']:8.2f}")

    if len(results) > 1:
        base = results[0]["at_tuned_threshold"]["roc_auc"]
        print("\n[INFO] Selisih ROC-AUC terhadap backend pertama "
              f"({results[0]['backend']}):")
        for r in results[1:]:
            delta = r["at_tuned_threshold"]["roc_auc"] - base
            speedup = results[0]["latency_ms_per_image"] / r["latency_ms_per_image"]
            print(f"    {r['backend']:16s} {delta:+.4f} AUC, {speedup:.2f}x lebih cepat")
    return 0


if __name__ == "__main__":
    sys.exit(main())
