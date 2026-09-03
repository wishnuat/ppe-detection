"""Classifier fatigue per-frame: crop wajah -> probabilitas kelas "fatigue".

Modul ini memegang *satu-satunya* definisi preprocessing yang dipakai bersama
oleh training (`scripts/train_fatigue.py`) dan inferensi. Itu disengaja:
skew antara transform training dan transform runtime adalah cara paling
umum sebuah classifier terlihat bagus di notebook lalu gagal di lapangan, dan
satu-satunya cara memastikan keduanya tidak melenceng adalah tidak
menuliskannya dua kali.

Checkpoint yang disimpan training membawa metadata lengkap (arsitektur, urutan
kelas, ukuran input, ambang keputusan hasil tuning di validation). Loader di
bawah membaca metadata itu alih-alih menebak, jadi mengganti arsitektur tidak
menuntut perubahan kode pemanggil.

Dua backend, mengikuti pola `src/backends.py`:

    torch      models/fatigue/fatigue_cls.pt — referensi akurasi.
    openvino   models/fatigue/fatigue_cls_openvino_model/ — jauh lebih cepat
               di CPU Intel, dipakai untuk deploy CCTV multi-wajah.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = Path(
    os.getenv("FATIGUE_ASSET_DIR", PROJECT_ROOT / "models" / "fatigue")
)
DEFAULT_CHECKPOINT = DEFAULT_MODEL_DIR / "fatigue_cls.pt"

# Urutan kelas dikunci di sini dan diverifikasi ulang saat load. Membaliknya
# tanpa sadar berarti "lelah" dan "segar" tertukar di seluruh sistem — bug yang
# diam-diam lolos semua test yang cuma mengecek akurasi.
CLASSES = ("nonfatigue", "fatigue")
FATIGUE_INDEX = 1

IMAGE_SIZE = 224
# Statistik ImageNet — semua backbone torchvision yang dipakai di-pretrain
# dengan normalisasi ini.
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASSIFIER_BACKENDS = ("torch", "openvino")

# Ambang default kalau checkpoint tidak membawa hasil tuning. 0.5 hanya optimal
# kalau kelasnya seimbang DAN biaya kedua jenis kesalahan sama — di sini
# keduanya tidak dijamin, jadi angka sebenarnya di-tune di validation set.
DEFAULT_THRESHOLD = 0.5


def preprocess_bgr(crop: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Crop wajah BGR (HWC uint8) -> tensor CHW float32 siap masuk model.

    Resize langsung ke persegi tanpa menjaga rasio aspek, sama seperti
    `transforms.Resize((size, size))` di jalur training. Letterbox terdengar
    lebih benar, tapi kalau dipakai di sini saja ia justru menciptakan skew
    yang ingin dihindari modul ini.
    """
    if crop.ndim == 2:
        crop = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return ((rgb - MEAN) / STD).transpose(2, 0, 1)


# Berapa kali lebih mahal melewatkan pekerja yang lelah (false negative)
# dibanding memicu satu alarm palsu (false positive). Dipakai sebagai default
# saat memilih ambang keputusan.
#
# 3:1 bukan angka ajaib — ia asumsi yang dinyatakan terbuka supaya bisa
# diperdebatkan dan diubah per lokasi (`scripts/tune_fatigue_threshold.py`).
# Yang penting: asumsinya BUKAN 1:1. Kriteria simetris seperti Youden's J
# memilih ambang yang membuang recall demi presisi, dan pada sistem peringatan
# keselamatan itu pertukaran yang arahnya salah.
#
# Angka ini bisa lebih agresif daripada yang terasa nyaman, karena classifier
# ini tidak pernah memicu alarm sendirian: ia hanya 30% dari skor fusi, dan
# masih harus melewati agregasi 60 detik plus histeresis sebelum sampai ke
# operator. Alarm palsu per-frame diserap di sana; recall yang hilang tidak
# bisa dipulihkan di mana pun.
DEFAULT_FN_COST = 3.0

THRESHOLD_CRITERIA = ("cost", "youden", "recall-floor", "f1")


def choose_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    criterion: str = "cost",
    fn_cost: float = DEFAULT_FN_COST,
    recall_floor: float = 0.93,
) -> tuple[float, dict]:
    """Pilih ambang keputusan dari probabilitas & label di *validation set*.

    Mengembalikan (ambang, diagnostik). Diagnostiknya ikut disimpan ke
    checkpoint supaya alasan pemilihan ambang bisa diaudit belakangan — angka
    ambang tanpa kriteria yang menghasilkannya adalah angka yang tidak bisa
    dipertanggungjawabkan.

    Kriteria:
        cost          minimalkan `fn_cost * FN + FP`. Default.
        youden        maksimalkan TPR - FPR. Simetris; disediakan sebagai
                      pembanding, bukan sebagai default.
        recall-floor  presisi tertinggi di antara ambang yang recall-nya masih
                      >= `recall_floor`. Dipakai kalau ada target recall yang
                      memang diwajibkan.
        f1            maksimalkan F1. Mengabaikan true negative sepenuhnya,
                      jadi ia cenderung terlalu sensitif pada data yang
                      sebagian besar normal — ada di sini untuk kelengkapan.
    """
    if criterion not in THRESHOLD_CRITERIA:
        raise ValueError(
            f"Kriteria '{criterion}' tidak dikenal. "
            f"Pilihan: {', '.join(THRESHOLD_CRITERIA)}"
        )

    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    targets = np.asarray(targets).reshape(-1).astype(int)

    # Kandidat ambang = titik tengah antara skor-skor unik yang berdekatan.
    # Mengevaluasi grid tetap (0.05, 0.10, ...) bisa melewatkan optimum yang
    # jatuh di antara dua kisi, dan jumlah kandidat di sini toh cuma sebanyak
    # gambar validasi.
    unique = np.unique(probs)
    candidates = np.unique(
        np.concatenate([[0.0], (unique[:-1] + unique[1:]) / 2.0, [1.0]])
        if len(unique) > 1 else np.array([0.0, 0.5, 1.0])
    )

    positives = int((targets == 1).sum())
    negatives = int((targets == 0).sum())
    rows = []
    for threshold in candidates:
        pred = probs >= threshold
        tp = int((pred & (targets == 1)).sum())
        fp = int((pred & (targets == 0)).sum())
        fn = positives - tp
        tn = negatives - fp
        recall = tp / positives if positives else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        fpr = fp / negatives if negatives else 0.0
        rows.append({
            "threshold": float(threshold),
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision,
            "youden": recall - fpr,
            "cost": fn_cost * fn + fp,
            "f1": (2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0),
        })

    if criterion == "cost":
        # Tie-break ke ambang TERTINGGI di antara yang biayanya sama: kalau dua
        # ambang salah pada gambar yang sama persis, yang lebih tinggi memberi
        # margin lebih lebar terhadap pergeseran distribusi di lapangan.
        best = min(rows, key=lambda r: (r["cost"], -r["threshold"]))
    elif criterion == "youden":
        best = max(rows, key=lambda r: (r["youden"], r["threshold"]))
    elif criterion == "f1":
        best = max(rows, key=lambda r: (r["f1"], r["threshold"]))
    else:
        eligible = [r for r in rows if r["recall"] >= recall_floor]
        if not eligible:
            # Tidak ada ambang yang memenuhi target; ambil recall tertinggi
            # yang bisa dicapai dan katakan apa adanya.
            best = max(rows, key=lambda r: (r["recall"], r["precision"]))
            best = dict(best, recall_floor_met=False)
        else:
            best = max(eligible, key=lambda r: (r["precision"], r["threshold"]))
            best = dict(best, recall_floor_met=True)

    diagnostics = {
        "criterion": criterion,
        "fn_cost": fn_cost if criterion == "cost" else None,
        "recall_floor": recall_floor if criterion == "recall-floor" else None,
        "chosen": {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in best.items()},
        "num_candidates": len(candidates),
    }
    return float(best["threshold"]), diagnostics


def softmax(logits: np.ndarray) -> np.ndarray:
    """Softmax numerik-stabil pada sumbu terakhir."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def build_backbone(arch: str, num_classes: int = 2, pretrained: bool = True):
    """Bangun backbone torchvision dengan kepala klasifikasi yang diganti.

    Dipisah dari kelas inference supaya `scripts/train_fatigue.py` memakai
    definisi arsitektur yang sama persis dengan yang nanti me-load bobotnya.
    """
    import torch.nn as nn
    from torchvision import models

    if not hasattr(models, arch):
        raise ValueError(
            f"Arsitektur '{arch}' tidak ada di torchvision.models. "
            "Contoh yang cocok untuk dataset kecil ini: mobilenet_v3_large, "
            "efficientnet_b0, resnet18."
        )
    weights = "DEFAULT" if pretrained else None
    model = getattr(models, arch)(weights=weights)

    # Nama atribut kepala klasifikasi berbeda-beda antar keluarga model;
    # dicari secara struktural supaya `--arch` bebas tanpa tabel manual.
    if hasattr(model, "classifier"):
        head = model.classifier
        if isinstance(head, nn.Sequential):
            for i in range(len(head) - 1, -1, -1):
                if isinstance(head[i], nn.Linear):
                    head[i] = nn.Linear(head[i].in_features, num_classes)
                    break
            else:
                raise ValueError(f"Tidak menemukan nn.Linear di classifier {arch}")
        elif isinstance(head, nn.Linear):
            model.classifier = nn.Linear(head.in_features, num_classes)
    elif hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif hasattr(model, "head") and isinstance(model.head, nn.Linear):
        model.head = nn.Linear(model.head.in_features, num_classes)
    else:
        raise ValueError(f"Tidak tahu cara mengganti kepala klasifikasi {arch}")
    return model


class FatigueClassifier:
    """Antarmuka inferensi. Subclass mengisi `_forward` (logits mentah)."""

    backend = "base"

    def __init__(self, threshold: float = DEFAULT_THRESHOLD, tta: bool = False) -> None:
        self.threshold = threshold
        # TTA horizontal-flip: dua kali biaya inferensi untuk sedikit stabilitas.
        # Default mati — di CCTV multi-wajah, latensi lebih mahal daripada
        # selisih akurasi yang didapat.
        self.tta = tta
        self.classes = CLASSES
        self.arch = "unknown"
        self.model_path = ""
        self.image_size = IMAGE_SIZE
        self.metadata: dict = {}

    # ---------- API publik ----------
    def predict_crop(self, crop: np.ndarray) -> float:
        """Probabilitas kelas 'fatigue' (0..1) untuk satu crop wajah."""
        return float(self.predict_batch([crop])[0])

    def predict_batch(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Probabilitas 'fatigue' untuk sekumpulan crop, dalam satu forward pass."""
        if not len(crops):
            return np.zeros(0, dtype=np.float32)

        batch = np.stack([preprocess_bgr(c, self.image_size) for c in crops])
        probs = softmax(self._forward(batch))[:, FATIGUE_INDEX]
        if self.tta:
            flipped = batch[:, :, :, ::-1].copy()
            probs = 0.5 * (probs + softmax(self._forward(flipped))[:, FATIGUE_INDEX])
        return probs.astype(np.float32)

    def is_fatigued(self, prob: float) -> bool:
        """Keputusan biner classifier ini, berdiri sendiri.

        **Tidak dipakai `FatiguePipeline`.** Fusi mengonsumsi probabilitas
        mentahnya sebagai bukti kontinu dan menggabungkannya dengan sinyal
        perilaku; membinerkannya lebih dulu hanya membuang informasi. Titik
        operasi sistem yang sebenarnya diatur oleh `FusionConfig.mild_at` dan
        kawan-kawannya, bukan oleh ambang ini.

        Yang dilayani method ini adalah pemakaian classifier secara terpisah —
        evaluasi, pelaporan, atau kalau seseorang ingin memakai modelnya tanpa
        pipeline. `scripts/tune_fatigue_threshold.py` yang menyetel angkanya.
        """
        return prob >= self.threshold

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def describe(self) -> str:
        return (f"{self.arch} @ {self.backend} "
                f"(threshold {self.threshold:.3f}, input {self.image_size}px)")


class TorchFatigueClassifier(FatigueClassifier):
    """Checkpoint PyTorch hasil `scripts/train_fatigue.py`."""

    backend = "torch"

    def __init__(
        self,
        checkpoint: str | Path | None = None,
        threshold: float | None = None,
        tta: bool = False,
        device: str = "cpu",
    ) -> None:
        import torch

        super().__init__(threshold=DEFAULT_THRESHOLD, tta=tta)
        path = Path(checkpoint or os.getenv("FATIGUE_MODEL_PATH", DEFAULT_CHECKPOINT))
        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint fatigue tidak ditemukan di {path}.\n"
                "Jalankan dulu:\n"
                "    python scripts/prepare_fatigue_dataset.py\n"
                "    python scripts/train_fatigue.py"
            )
        self.model_path = str(path)
        self.device = device

        ckpt = torch.load(path, map_location=device, weights_only=False)
        self.metadata = {k: v for k, v in ckpt.items() if k != "state_dict"}
        self.arch = ckpt.get("arch", "mobilenet_v3_large")
        self.image_size = int(ckpt.get("image_size", IMAGE_SIZE))
        classes = tuple(ckpt.get("classes", CLASSES))
        if classes != CLASSES:
            raise ValueError(
                f"Urutan kelas checkpoint {classes} tidak sama dengan {CLASSES}. "
                "Memakainya akan menukar arti 'lelah' dan 'segar'."
            )
        # Ambang eksplisit dari pemanggil menang atas hasil tuning checkpoint.
        self.threshold = (
            threshold if threshold is not None
            else float(ckpt.get("threshold", DEFAULT_THRESHOLD))
        )

        model = build_backbone(self.arch, num_classes=len(CLASSES), pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval().to(device)
        self._model = model
        self._torch = torch

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            tensor = torch.from_numpy(batch).to(self.device)
            return self._model(tensor).cpu().numpy()


class OpenVINOFatigueClassifier(FatigueClassifier):
    """IR OpenVINO hasil `scripts/export_fatigue.py`.

    IR-nya diexport dengan sumbu batch dinamis (`[?,3,224,224]`), jadi seluruh
    batch bisa diumpankan sekaligus tanpa rekompilasi. Itu penting: menjalankan
    wajah satu per satu mengorbankan sebagian besar keuntungan OpenVINO — pada
    mesin uji, batch 8 sekaligus jatuh ke 7,5 ms/gambar sedangkan batch 1
    berulang butuh 12,5 ms/gambar.
    """

    backend = "openvino"

    def __init__(
        self,
        model_dir: str | Path | None = None,
        threshold: float | None = None,
        tta: bool = False,
        device: str | None = None,
        int8: bool = False,
    ) -> None:
        import json

        import openvino as ov

        super().__init__(threshold=DEFAULT_THRESHOLD, tta=tta)
        suffix = "_int8" if int8 else ""
        directory = Path(
            model_dir
            or os.getenv("FATIGUE_OPENVINO_DIR")
            or DEFAULT_MODEL_DIR / f"fatigue_cls{suffix}_openvino_model"
        )
        xml = directory / "fatigue_cls.xml"
        if not xml.exists():
            raise FileNotFoundError(
                f"IR OpenVINO tidak ditemukan di {xml}.\n"
                "Jalankan dulu: python scripts/export_fatigue.py"
            )
        self.model_path = str(xml)
        self.device = device or os.getenv("OPENVINO_DEVICE", "CPU")

        meta_file = directory / "metadata.json"
        if meta_file.exists():
            self.metadata = json.loads(meta_file.read_text(encoding="utf-8"))
            self.arch = self.metadata.get("arch", "unknown")
            self.image_size = int(self.metadata.get("image_size", IMAGE_SIZE))
            if threshold is None:
                self.threshold = float(self.metadata.get("threshold", DEFAULT_THRESHOLD))
        if threshold is not None:
            self.threshold = threshold

        core = ov.Core()
        self._compiled = core.compile_model(core.read_model(xml), self.device)
        self._output = self._compiled.output(0)

    def _forward(self, batch: np.ndarray) -> np.ndarray:
        return self._compiled([batch])[self._output]


def build_classifier(
    backend: str | None = None,
    threshold: float | None = None,
    tta: bool = False,
    **kwargs,
) -> FatigueClassifier:
    """Bangun classifier sesuai backend (default env FATIGUE_BACKEND)."""
    name = (backend or os.getenv("FATIGUE_BACKEND", "torch")).strip().lower()
    if name in ("openvino-int8", "int8"):
        return OpenVINOFatigueClassifier(threshold=threshold, tta=tta, int8=True, **kwargs)
    if name in ("openvino", "ov"):
        return OpenVINOFatigueClassifier(threshold=threshold, tta=tta, **kwargs)
    if name in ("torch", "pytorch", "local"):
        return TorchFatigueClassifier(threshold=threshold, tta=tta, **kwargs)
    raise ValueError(
        f"Backend classifier '{backend}' tidak dikenal. "
        f"Pilihan: {', '.join(CLASSIFIER_BACKENDS)}, openvino-int8"
    )
