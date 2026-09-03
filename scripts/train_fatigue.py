"""Latih classifier fatigue (transfer learning) di atas dataset hasil prepare.

    python scripts/prepare_fatigue_dataset.py
    python scripts/train_fatigue.py --arch mobilenet_v3_large --epochs 30

Hasil: models/fatigue/fatigue_cls.pt  (+ laporan metrik di outputs/fatigue/)

Dataset ini kecil (1.540 gambar train) dan sudah dipecah per identitas, jadi
tiga hal menentukan apakah hasilnya berguna atau sekadar angka bagus:

1. Augmentasi agresif, tapi yang masuk akal untuk wajah.
   RandomResizedCrop + flip + jitter warna + rotasi kecil + RandomErasing.
   Vertical flip dan rotasi besar TIDAK dipakai — wajah terbalik bukan
   distribusi yang akan pernah ditemui kamera CCTV, dan melatihnya cuma
   memboroskan kapasitas model.

2. Ambang keputusan di-tune, bukan diasumsikan 0.5.
   Ambang dipilih di validation set (maksimum Youden's J) lalu disimpan ke
   checkpoint. Test set tidak pernah menyentuh pemilihan ambang — itu akan
   membuat angka test-nya optimistis.

3. Model terbaik dipilih pakai ROC-AUC validation, bukan akurasi.
   Akurasi bergantung pada ambang yang belum ditentukan pada saat itu; AUC
   tidak, jadi ia ukuran yang benar untuk membandingkan epoch.

EMA (exponential moving average) bobot dipakai karena pada dataset sekecil ini
bobot per-epoch berayun cukup keras; rata-rata bergeraknya konsisten lebih baik
dan gratis secara komputasi.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.classifier import (  # noqa: E402
    CLASSES,
    FATIGUE_INDEX,
    IMAGE_SIZE,
    MEAN,
    STD,
    build_backbone,
)

DEFAULT_DATA = PROJECT_ROOT / "datasets" / "fatigue"
DEFAULT_OUT = PROJECT_ROOT / "models" / "fatigue" / "fatigue_cls.pt"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "outputs" / "fatigue"


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(image_size: int):
    """(transform_train, transform_eval).

    `transform_eval` sengaja dibuat setara dengan `classifier.preprocess_bgr`:
    resize langsung ke persegi, normalisasi ImageNet, tanpa center-crop.
    Kalau salah satunya diubah, ubah keduanya.
    """
    from torchvision import transforms

    normalize = transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist())
    train = transforms.Compose([
        transforms.RandomResizedCrop(
            image_size, scale=(0.65, 1.0), ratio=(0.85, 1.18)
        ),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(brightness=0.35, contrast=0.35,
                                    saturation=0.25, hue=0.03)], p=0.8
        ),
        # CCTV malam & lampu pabrik bikin frame nyaris monokrom — model tidak
        # boleh bergantung pada warna kulit atau suhu warna lampu.
        transforms.RandomGrayscale(p=0.15),
        transforms.RandomRotation(12),
        transforms.ToTensor(),
        normalize,
        # Meniru oklusi nyata: masker, tangan mengusap mata, helm yang turun.
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), value="random"),
    ])
    evaluate = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train, evaluate


def _fixed_order_image_folder():
    """ImageFolder dengan urutan kelas dipaksa = `classifier.CLASSES`.

    ImageFolder memberi indeks berdasarkan urutan alfabetis nama folder, yang
    di sini berarti fatigue=0, nonfatigue=1 — kebalikan dari konvensi proyek.
    Membiarkannya berarti seluruh sistem melaporkan 'lelah' untuk orang segar,
    dan tidak ada satu pun metrik akurasi yang akan menunjukkan gejalanya.
    Jadi `find_classes` di-override, bukan sekadar dicek.

    Dibungkus fungsi (bukan kelas modul-level biasa) supaya `torchvision`
    tetap di-import malas, tapi hasilnya di-cache di global — DataLoader worker
    di Windows memakai spawn dan harus bisa mem-pickle kelas dataset lewat
    nama modul + nama kelasnya.
    """
    global _FIXED_FOLDER_CLS
    if _FIXED_FOLDER_CLS is not None:
        return _FIXED_FOLDER_CLS

    from torchvision.datasets import ImageFolder

    class FixedOrderImageFolder(ImageFolder):
        def find_classes(self, directory):
            missing = [c for c in CLASSES if not (Path(directory) / c).is_dir()]
            if missing:
                raise SystemExit(
                    f"[ERROR] Folder kelas tidak ada di {directory}: {missing}. "
                    "Jalankan ulang scripts/prepare_fatigue_dataset.py"
                )
            return list(CLASSES), {c: i for i, c in enumerate(CLASSES)}

    globals()["FixedOrderImageFolder"] = FixedOrderImageFolder
    _FIXED_FOLDER_CLS = FixedOrderImageFolder
    return FixedOrderImageFolder


_FIXED_FOLDER_CLS = None


def make_loader(root: Path, transform, batch_size: int, shuffle: bool, workers: int):
    from torch.utils.data import DataLoader

    ds = _fixed_order_image_folder()(str(root), transform=transform)
    assert tuple(ds.classes) == CLASSES, ds.classes
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
        pin_memory=False, drop_last=False,
        persistent_workers=workers > 0,
    )


class ModelEMA:
    """Rata-rata bergerak eksponensial atas bobot model.

    Decay dinaikkan bertahap (`warmup`) supaya bayangan EMA tidak tertahan
    lama di bobot inisialisasi acak pada epoch-epoch pertama.
    """

    def __init__(self, model, decay: float = 0.999, warmup: int = 200) -> None:
        self.module = deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.updates = 0

    def update(self, model) -> None:
        import torch

        self.updates += 1
        d = self.decay * (1 - np.exp(-self.updates / self.warmup))
        with torch.no_grad():
            msd = model.state_dict()
            for k, v in self.module.state_dict().items():
                if v.dtype.is_floating_point:
                    v.mul_(d).add_(msd[k].detach(), alpha=1 - d)
                else:
                    # Buffer integer (mis. num_batches_tracked) tidak bisa
                    # dirata-rata; disalin apa adanya.
                    v.copy_(msd[k])


def evaluate_probs(model, loader, device: str) -> tuple[np.ndarray, np.ndarray]:
    """(probabilitas kelas fatigue, label sebenarnya) untuk seluruh loader."""
    import torch

    model.eval()
    probs, targets = [], []
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            probs.append(torch.softmax(logits, dim=1)[:, FATIGUE_INDEX].cpu().numpy())
            targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def metrics_at(probs: np.ndarray, targets: np.ndarray, threshold: float) -> dict:
    """Metrik klasifikasi biner pada satu ambang, plus AUC yang bebas ambang."""
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
    # AUC tidak terdefinisi kalau split cuma berisi satu kelas.
    single_class = len(np.unique(targets)) < 2
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((pred == targets).mean()), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "specificity": round(float(specificity), 4),
        "f1": round(float(f1), 4),
        "roc_auc": None if single_class else round(float(roc_auc_score(targets, probs)), 4),
        "pr_auc": None if single_class else round(
            float(average_precision_score(targets, probs)), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "support": {"nonfatigue": int((targets == 0).sum()), "fatigue": int((targets == 1).sum())},
    }


def tune_threshold(probs: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    """Ambang yang memaksimalkan Youden's J (= TPR - FPR) di validation set.

    Youden dipilih, bukan max-F1: F1 mengabaikan true negative, sehingga pada
    sistem monitoring yang sebagian besar frame-nya normal ia cenderung
    memilih ambang terlalu rendah dan membanjiri operator dengan alarm palsu.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(targets, probs)
    j = tpr - fpr
    best = int(np.argmax(j))
    # roc_curve menaruh threshold=inf di indeks 0 sebagai titik (0,0).
    thr = float(thresholds[best])
    if not np.isfinite(thr):
        thr = 1.0
    return thr, float(j[best])


def main() -> int:
    import torch
    import torch.nn as nn

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    ap.add_argument("--arch", type=str, default="mobilenet_v3_large",
                    help="Backbone torchvision. mobilenet_v3_large (default, "
                         "cepat di CPU), efficientnet_b0, resnet18.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4,
                    help="LR kepala klasifikasi. Backbone memakai lr/10.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader workers. 0 paling aman di Windows.")
    ap.add_argument("--patience", type=int, default=10,
                    help="Berhenti kalau ROC-AUC val tidak membaik sekian epoch.")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--device", type=str,
                    default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.data / "train").is_dir():
        raise SystemExit(
            f"[ERROR] {args.data}/train tidak ada. "
            "Jalankan dulu: python scripts/prepare_fatigue_dataset.py"
        )

    seed_everything(args.seed)
    device = args.device
    t_train, t_eval = build_transforms(args.image_size)
    train_loader = make_loader(args.data / "train", t_train, args.batch_size, True, args.workers)
    val_loader = make_loader(args.data / "val", t_eval, args.batch_size, False, args.workers)
    test_loader = make_loader(args.data / "test", t_eval, args.batch_size, False, args.workers)

    print(f"[INFO] arch={args.arch} device={device} "
          f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)}")

    model = build_backbone(args.arch, num_classes=len(CLASSES), pretrained=True).to(device)

    # LR diskriminatif: kepala klasifikasi baru diinisialisasi acak dan butuh
    # belajar cepat, sedangkan fitur backbone hasil pretrain ImageNet sudah
    # bagus dan mudah rusak kalau digerakkan sekeras itu.
    head_names = {"classifier", "fc", "head"}
    head_params, backbone_params = [], []
    for name, param in model.named_parameters():
        (head_params if name.split(".")[0] in head_names else backbone_params).append(param)
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr / 10},
         {"params": head_params, "lr": args.lr}],
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    ema = None if args.no_ema else ModelEMA(model)

    best_auc, best_state, best_epoch, stale = -1.0, None, -1, 0
    history = []
    t0 = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += loss.item() * x.size(0)
            seen += x.size(0)
        scheduler.step()

        eval_model = ema.module if ema is not None else model
        probs, targets = evaluate_probs(eval_model, val_loader, device)
        val = metrics_at(probs, targets, 0.5)
        auc = val["roc_auc"] or 0.0
        history.append({"epoch": epoch, "train_loss": round(running / seen, 4), **val})
        print(f"  epoch {epoch:3d}/{args.epochs}  loss={running / seen:.4f}  "
              f"val_auc={auc:.4f}  val_acc={val['accuracy']:.4f}  "
              f"({time.perf_counter() - t0:.0f}s)")

        if auc > best_auc:
            best_auc, best_epoch, stale = auc, epoch, 0
            best_state = deepcopy(eval_model.state_dict())
        else:
            stale += 1
            if stale >= args.patience:
                print(f"[INFO] Early stop di epoch {epoch} "
                      f"(ROC-AUC val tidak membaik {args.patience} epoch).")
                break

    if best_state is None:
        raise SystemExit("[ERROR] Training tidak menghasilkan bobot.")

    # ---- ambang di-tune di VAL, metrik akhir diukur di TEST ----
    model.load_state_dict(best_state)
    val_probs, val_targets = evaluate_probs(model, val_loader, device)
    threshold, youden = tune_threshold(val_probs, val_targets)
    val_metrics = metrics_at(val_probs, val_targets, threshold)

    test_probs, test_targets = evaluate_probs(model, test_loader, device)
    test_metrics = metrics_at(test_probs, test_targets, threshold)
    test_at_half = metrics_at(test_probs, test_targets, 0.5)

    print(f"\n[INFO] Epoch terbaik {best_epoch} (val ROC-AUC {best_auc:.4f})")
    print(f"[INFO] Ambang hasil tuning di val: {threshold:.4f} (Youden J {youden:.4f})")
    print(f"[VAL ] acc={val_metrics['accuracy']:.4f} f1={val_metrics['f1']:.4f} "
          f"auc={val_metrics['roc_auc']}")
    print(f"[TEST] acc={test_metrics['accuracy']:.4f} f1={test_metrics['f1']:.4f} "
          f"prec={test_metrics['precision']:.4f} rec={test_metrics['recall']:.4f} "
          f"auc={test_metrics['roc_auc']} pr_auc={test_metrics['pr_auc']}")
    cm = test_metrics["confusion_matrix"]
    print(f"[TEST] confusion  tn={cm['tn']} fp={cm['fp']} fn={cm['fn']} tp={cm['tp']}")

    manifest_path = args.data / "manifest.json"
    dataset_meta = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists() else {}
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "arch": args.arch,
        "classes": list(CLASSES),
        "image_size": args.image_size,
        "threshold": threshold,
        "normalization": {"mean": MEAN.tolist(), "std": STD.tolist()},
        "best_epoch": best_epoch,
        "val_roc_auc": best_auc,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": dataset_meta,
        "args": vars(args) | {"data": str(args.data), "out": str(args.out),
                              "report_dir": str(args.report_dir)},
    }, args.out)
    print(f"[OK] Checkpoint: {args.out}")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "arch": args.arch,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration_sec": round(time.perf_counter() - t0, 1),
        "best_epoch": best_epoch,
        "threshold_tuned_on_val": threshold,
        "val": val_metrics,
        "test": test_metrics,
        "test_at_threshold_0.5": test_at_half,
        "dataset": dataset_meta,
        "history": history,
    }
    report_path = args.report_dir / f"report_{args.arch}.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        args.report_dir / f"probs_{args.arch}.npz",
        val_probs=val_probs, val_targets=val_targets,
        test_probs=test_probs, test_targets=test_targets,
    )
    print(f"[OK] Laporan: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
