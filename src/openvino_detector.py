"""Backend inference OpenVINO — jalur cepat untuk CPU / iGPU Intel.

Kenapa ada backend ini: training YOLOv8 di CPU sudah berat, dan inference
PyTorch di CPU juga jauh dari realtime. OpenVINO mengompilasi model ke
representasi IR yang dioptimasi untuk hardware Intel, jadi FPS-nya naik
signifikan tanpa mengubah arsitektur model. Ini juga jalur yang akan dipakai
saat sistem dipasang di edge box (NUC / mini-PC Intel di pos satpam atau
kabin unit), karena runtime-nya tidak butuh PyTorch sama sekali.

Preprocessing (letterbox) dan postprocessing (NMS) ditulis manual di sini
memakai numpy + OpenCV supaya proses serving benar-benar bebas torch —
image Docker jadi jauh lebih kecil dan cold start lebih cepat.

Pakai:
    from src.openvino_detector import OpenVINODetector
    det = OpenVINODetector(device="CPU")      # atau "GPU" / "AUTO"
    result = det.predict_frame(frame)
"""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import yaml
from dotenv import load_dotenv

from src.detector import Detection, DetectionResult, PPEDetector, classify_label

load_dotenv()

DEFAULT_MODEL_DIR = "models/best_openvino_model"
DEFAULT_INT8_DIR = "models/best_int8_openvino_model"

# Nilai pad standar YOLO. Harus sama dengan yang dipakai saat training,
# kalau tidak, box hasil deteksi akan bergeser.
PAD_COLOR = (114, 114, 114)


def letterbox(
    img: np.ndarray, new_shape: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize dengan menjaga aspect ratio, sisanya di-pad abu-abu.

    Kembalikan (gambar, rasio skala, (pad_kiri, pad_atas)) supaya koordinat
    box bisa dipetakan balik ke gambar asli.
    """
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    unpad_w, unpad_h = int(round(w * r)), int(round(h * r))

    if (w, h) != (unpad_w, unpad_h):
        # INTER_LINEAR, bukan INTER_AREA, meski sedang mengecilkan gambar:
        # harus sama persis dengan letterbox Ultralytics supaya hasil deteksi
        # identik dengan backend PyTorch.
        img = cv2.resize(img, (unpad_w, unpad_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = new_shape[1] - unpad_w, new_shape[0] - unpad_h
    left, top = dw // 2, dh // 2
    img = cv2.copyMakeBorder(
        img, top, dh - top, left, dw - left, cv2.BORDER_CONSTANT, value=PAD_COLOR
    )
    return img, r, (left, top)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    """Non-maximum suppression sederhana di numpy. boxes = [N, 4] xyxy."""
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        order = rest[iou <= iou_thres]
    return keep


class OpenVINODetector(PPEDetector):
    """Drop-in replacement `PPEDetector` yang inference-nya lewat OpenVINO.

    Interface-nya sama persis (predict_frame / render / draw_hud / predict_image
    / predict_video / predict_webcam diwarisi dari base class), jadi CLI, API,
    dan Streamlit tidak perlu tahu backend mana yang dipakai.
    """

    def __init__(
        self,
        model_dir: str | None = None,
        device: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        int8: bool = False,
        max_det: int = 300,
    ) -> None:
        # Sengaja tidak memanggil super().__init__(): tidak ada file .pt.
        if model_dir is None:
            env_dir = os.getenv("OPENVINO_MODEL_DIR")
            model_dir = env_dir or (DEFAULT_INT8_DIR if int8 else DEFAULT_MODEL_DIR)
        self.model_dir = Path(model_dir)
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model OpenVINO tidak ditemukan di {self.model_dir}. "
                "Jalankan `python scripts/export_openvino.py` dulu."
            )

        xmls = sorted(self.model_dir.glob("*.xml"))
        if not xmls:
            raise FileNotFoundError(f"Tidak ada file .xml di {self.model_dir}")
        self.xml_path = xmls[0]
        self.model_path = str(self.xml_path)

        self.device = (device or os.getenv("OPENVINO_DEVICE", "CPU")).upper()
        self.conf = conf if conf is not None else float(os.getenv("CONF_THRESHOLD", "0.35"))
        self.iou = iou if iou is not None else float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.max_det = max_det
        self.enabled_categories: set[str] | None = None

        meta = self._load_metadata()
        self.class_names = {int(k): v for k, v in (meta.get("names") or {}).items()}
        self.imgsz = self._resolve_imgsz(meta.get("imgsz"))

        import openvino as ov

        core = ov.Core()
        if self.device not in core.available_devices and self.device != "AUTO":
            raise RuntimeError(
                f"Device OpenVINO '{self.device}' tidak tersedia. "
                f"Pilihan: {', '.join(core.available_devices)} (atau AUTO)"
            )
        model = core.read_model(self.xml_path)
        # LATENCY = optimalkan waktu per frame (streaming kamera), bukan throughput
        # batch besar. Ini yang relevan untuk deteksi realtime.
        self.compiled = core.compile_model(
            model, self.device, {"PERFORMANCE_HINT": "LATENCY"}
        )
        self.output_port = self.compiled.output(0)
        self.infer_request = self.compiled.create_infer_request()

    # ---------- setup helpers ----------
    def _load_metadata(self) -> dict:
        """Baca metadata.yaml hasil export Ultralytics (berisi names & imgsz)."""
        meta_path = self.model_dir / "metadata.yaml"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"metadata.yaml tidak ada di {self.model_dir}. "
                "Export ulang lewat `python scripts/export_openvino.py`."
            )
        return yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}

    @staticmethod
    def _resolve_imgsz(raw) -> tuple[int, int]:
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            return int(raw[0]), int(raw[1])
        if isinstance(raw, (int, float)):
            return int(raw), int(raw)
        return 640, 640

    # ---------- inference ----------
    def predict_frame(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        out = DetectionResult(width=w, height=h)

        blob, ratio, (pad_x, pad_y) = self._preprocess(frame)
        raw = self.infer_request.infer({0: blob})[self.output_port]

        for label, conf, bbox in self._postprocess(raw, ratio, pad_x, pad_y, w, h):
            category, is_violation = classify_label(label)
            out.detections.append(
                Detection(
                    label=label,
                    category=category,
                    confidence=round(conf, 4),
                    bbox=bbox,
                    is_violation=is_violation,
                )
            )
        return self._finalize(out)

    def _preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        img, ratio, pad = letterbox(frame, self.imgsz)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        return np.ascontiguousarray(blob), ratio, pad

    def _postprocess(
        self,
        raw: np.ndarray,
        ratio: float,
        pad_x: int,
        pad_y: int,
        orig_w: int,
        orig_h: int,
    ) -> list[tuple[str, float, list[int]]]:
        """Ubah output mentah YOLOv8 [1, 4+nc, N] jadi daftar deteksi."""
        pred = np.squeeze(raw, axis=0).T  # -> [N, 4+nc]
        if pred.size == 0:
            return []

        boxes_xywh = pred[:, :4]
        class_scores = pred[:, 4:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

        # Pakai ambang terendah di antara conf global & override per kategori;
        # penyaringan final per kategori dikerjakan `_finalize`.
        keep_mask = scores >= self.detection_floor
        if not keep_mask.any():
            return []
        boxes_xywh = boxes_xywh[keep_mask]
        scores = scores[keep_mask]
        class_ids = class_ids[keep_mask]

        # Batasi kandidat sebelum NMS supaya biaya NMS tidak meledak di frame ramai.
        if scores.shape[0] > self.max_det * 10:
            top = np.argpartition(-scores, self.max_det * 10)[: self.max_det * 10]
            boxes_xywh, scores, class_ids = boxes_xywh[top], scores[top], class_ids[top]

        cx, cy, bw, bh = boxes_xywh.T
        boxes = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        # NMS per-kelas dikerjakan sekali jalan: geser tiap kelas ke "kanal"
        # koordinat sendiri supaya box antar kelas tidak pernah saling menekan.
        offset = class_ids[:, None] * (max(self.imgsz) + 1)
        keep = nms(boxes + offset, scores, self.iou)[: self.max_det]

        results: list[tuple[str, float, list[int]]] = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            x1 = int(np.clip((x1 - pad_x) / ratio, 0, orig_w - 1))
            y1 = int(np.clip((y1 - pad_y) / ratio, 0, orig_h - 1))
            x2 = int(np.clip((x2 - pad_x) / ratio, 0, orig_w - 1))
            y2 = int(np.clip((y2 - pad_y) / ratio, 0, orig_h - 1))
            label = self.class_names.get(int(class_ids[i]), str(class_ids[i]))
            results.append((label, float(scores[i]), [x1, y1, x2, y2]))
        return results
