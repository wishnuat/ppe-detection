"""PPE Detection inference wrapper on top of Ultralytics YOLOv8."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Kategori kepatuhan yang dilaporkan ke user.
PPE_CLASSES = [
    "helmet",
    "glasses",
    "mask",
    "glove",
    "shoes",
    "vest",
    "ear_protection",
    "harness",
]

# `person` bukan APD, tapi tetap bisa di-toggle di UI supaya box-nya
# bisa disembunyikan.
SELECTABLE_CATEGORIES = PPE_CLASSES + ["person"]

# Peta label mentah model (dataset wishnus-workspace/ppe-detection-hyeuz-6cijw v2,
# 17 kelas) ke (kategori kepatuhan, apakah ini pelanggaran).
# `person` ikut dideteksi tapi bukan APD, jadi tidak masuk PPE_CLASSES.
RAW_LABEL_MAP: dict[str, tuple[str, bool]] = {
    "head_helmet": ("helmet", False),
    "head_nohelmet": ("helmet", True),
    "glasses": ("glasses", False),
    "No_Glasses": ("glasses", True),
    "face_mask": ("mask", False),
    "face_nomask": ("mask", True),
    "hand_glove": ("glove", False),
    "hand_noglove": ("glove", True),
    "shoes": ("shoes", False),
    "boots": ("shoes", False),
    "Barefoots": ("shoes", True),
    "Sandals": ("shoes", True),
    "vest": ("vest", False),
    "Ear-protection": ("ear_protection", False),
    "No_Ear-Protection": ("ear_protection", True),
    "Harness": ("harness", False),
    "person": ("person", False),
}


def classify_label(raw: str) -> tuple[str, bool]:
    """Kembalikan (kategori, is_violation) untuk satu label mentah model.

    Label di luar peta ditebak dari prefix `no`/`No_` supaya model hasil
    retraining dengan kelas tambahan tetap tertangani.
    """
    if raw in RAW_LABEL_MAP:
        return RAW_LABEL_MAP[raw]
    low = raw.lower()
    for prefix in ("no_", "no-", "non_", "nomor_"):
        if low.startswith(prefix):
            return low[len(prefix):], True
    return low, False

COLOR_OK = (0, 200, 0)          # BGR — hijau
COLOR_VIOLATION = (0, 0, 255)   # BGR — merah


@dataclass
class Detection:
    label: str          # label mentah dari model, mis. "head_nohelmet"
    category: str       # kategori kepatuhan, mis. "helmet"
    confidence: float
    bbox: list[int]  # [x1, y1, x2, y2]
    is_violation: bool

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DetectionResult:
    detections: list[Detection] = field(default_factory=list)
    compliance: dict[str, str] = field(default_factory=dict)
    width: int = 0
    height: int = 0

    def to_dict(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "compliance": self.compliance,
            "detections": [d.to_dict() for d in self.detections],
        }


class PPEDetector:
    def __init__(
        self,
        model_path: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
    ) -> None:
        self.model_path = model_path or os.getenv("MODEL_PATH", "models/best.pt")
        self.conf = conf if conf is not None else float(os.getenv("CONF_THRESHOLD", "0.35"))
        self.iou = iou if iou is not None else float(os.getenv("IOU_THRESHOLD", "0.45"))

        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model tidak ditemukan di {self.model_path}. "
                "Jalankan `python scripts/download_dataset.py` lalu "
                "`python scripts/train.py` dulu."
            )
        self.model = YOLO(self.model_path)
        self.class_names = self._resolve_names(self.model.names)

        # None = semua kategori aktif. Set ke iterable kategori untuk memfilter.
        self.enabled_categories: set[str] | None = None

    @staticmethod
    def _resolve_names(names: dict[int, str] | list[str]) -> dict[int, str]:
        if isinstance(names, list):
            names = {i: n for i, n in enumerate(names)}
        return dict(names)

    # ---------- core ----------
    def predict_frame(self, frame: np.ndarray) -> DetectionResult:
        results = self.model.predict(
            source=frame, conf=self.conf, iou=self.iou, verbose=False
        )
        r = results[0]
        h, w = frame.shape[:2]
        out = DetectionResult(width=w, height=h)

        if r.boxes is None or len(r.boxes) == 0:
            return self._finalize(out)

        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = self.class_names.get(cls_id, str(cls_id))
            category, is_violation = classify_label(label)
            conf = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            out.detections.append(
                Detection(
                    label=label,
                    category=category,
                    confidence=round(conf, 4),
                    bbox=[x1, y1, x2, y2],
                    is_violation=is_violation,
                )
            )
        return self._finalize(out)

    def _finalize(self, out: DetectionResult) -> DetectionResult:
        """Terapkan filter kategori lalu hitung ulang compliance.

        Dipanggil semua backend supaya CLI, API, dan UI konsisten.
        """
        allowed = self.enabled_categories
        if allowed is not None:
            allowed = set(allowed)
            out.detections = [d for d in out.detections if d.category in allowed]

        out.compliance = self._compute_compliance(out.detections)
        if allowed is not None:
            out.compliance = {k: v for k, v in out.compliance.items() if k in allowed}
        return out

    @staticmethod
    def _compute_compliance(detections: Iterable[Detection]) -> dict[str, str]:
        dets = list(detections)
        violated = {d.category for d in dets if d.is_violation}
        present = {d.category for d in dets if not d.is_violation}
        status = {}
        for ppe in PPE_CLASSES:
            if ppe in violated:
                status[ppe] = "PELANGGARAN"
            elif ppe in present:
                status[ppe] = "TERDETEKSI"
            else:
                status[ppe] = "TIDAK TERDETEKSI"
        return status

    # ---------- rendering ----------
    @staticmethod
    def render(frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        out = frame.copy()
        for d in result.detections:
            x1, y1, x2, y2 = d.bbox
            color = COLOR_VIOLATION if d.is_violation else COLOR_OK
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            text = f"{d.label} {d.confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                out, text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out

    # ---------- convenience ----------
    def predict_image(self, image_path: str) -> tuple[np.ndarray, DetectionResult]:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Gagal membaca image: {image_path}")
        res = self.predict_frame(img)
        return self.render(img, res), res

    def predict_video(
        self,
        video_path: str,
        output_path: str | None = None,
    ) -> str:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Gagal membuka video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_path = output_path or f"outputs/annotated_{Path(video_path).stem}.mp4"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = self.predict_frame(frame)
                writer.write(self.render(frame, res))
        finally:
            cap.release()
            writer.release()
        return out_path

    @staticmethod
    def draw_hud(frame: np.ndarray, result: DetectionResult, fps: float | None = None) -> np.ndarray:
        """Gambar panel compliance status di pojok kiri atas."""
        out = frame
        pad, line_h = 8, 18
        rows = [p for p, s in result.compliance.items() if s != "TIDAK TERDETEKSI"]
        height = line_h * (len(rows) + 1) + pad * 2

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (250, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        header = "PPE COMPLIANCE"
        if fps is not None:
            header += f"  {fps:4.1f} fps"
        cv2.putText(out, header, (pad, pad + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        for i, ppe in enumerate(rows, start=1):
            status = result.compliance[ppe]
            color = COLOR_VIOLATION if status == "PELANGGARAN" else COLOR_OK
            cv2.putText(out, f"{ppe}: {status}", (pad, pad + 12 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        return out

    def predict_webcam(self, camera_index: int = 0, save_video: str | None = None) -> None:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            raise RuntimeError(f"Gagal membuka kamera index {camera_index}")

        writer = None
        if save_video:
            Path(save_video).parent.mkdir(parents=True, exist_ok=True)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            writer = cv2.VideoWriter(
                save_video, cv2.VideoWriter_fourcc(*"mp4v"), 15.0, (w, h)
            )

        print("[INFO] Tekan 'q' di window untuk berhenti, 's' untuk simpan snapshot.")
        prev_violations: set[str] = set()
        t_prev = time.perf_counter()
        fps = 0.0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                res = self.predict_frame(frame)

                now = time.perf_counter()
                dt = now - t_prev
                t_prev = now
                if dt > 0:
                    fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

                annotated = self.draw_hud(self.render(frame, res), res, fps)

                # Hanya cetak saat status pelanggaran berubah, bukan tiap frame.
                violations = {p for p, s in res.compliance.items() if s == "PELANGGARAN"}
                for ppe in sorted(violations - prev_violations):
                    print(f"[ALERT] {ppe}: PELANGGARAN")
                for ppe in sorted(prev_violations - violations):
                    print(f"[OK]    {ppe}: pelanggaran selesai")
                prev_violations = violations

                if writer is not None:
                    writer.write(annotated)

                cv2.imshow("PPE Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    snap = Path("outputs") / f"snapshot_{int(now)}.jpg"
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap), annotated)
                    print(f"[OK] Snapshot disimpan: {snap}")
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            # Jangan biarkan cleanup menutupi error asli (mis. OpenCV headless
            # tanpa dukungan GUI).
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
