"""Backend inference via Roboflow Serverless API.

Berguna untuk dua hal:
1. Demo cepat sebelum weights lokal selesai dilatih.
2. Pembanding (baseline) terhadap model lokal hasil `scripts/train.py`.

Interface-nya identik dengan `PPEDetector`, jadi CLI / API / Streamlit bisa
memakainya tanpa perubahan lain. Untuk deployment lapangan tetap pakai
`PPEDetector` (offline, tanpa network round-trip per frame).
"""
from __future__ import annotations

import base64
import os

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

from src.detector import Detection, DetectionResult, PPEDetector, classify_label

load_dotenv()

SERVERLESS_URL = "https://serverless.roboflow.com"


class RoboflowDetector(PPEDetector):
    """Drop-in replacement PPEDetector yang inference-nya lewat Roboflow."""

    def __init__(
        self,
        model_id: str | None = None,
        api_key: str | None = None,
        conf: float | None = None,
        iou: float | None = None,
        timeout: float = 30.0,
    ) -> None:
        # Sengaja tidak memanggil super().__init__(): tidak ada model lokal.
        self.api_key = api_key or os.getenv("ROBOFLOW_API_KEY")
        if not self.api_key:
            raise ValueError("ROBOFLOW_API_KEY belum di-set di .env")

        project = os.getenv("ROBOFLOW_PROJECT", "ppe-detection-hyeuz-6cijw")
        version = os.getenv("ROBOFLOW_VERSION", "2")
        self.model_id = model_id or f"{project}/{version}"
        self.model_path = f"roboflow:{self.model_id}"

        self.conf = conf if conf is not None else float(os.getenv("CONF_THRESHOLD", "0.35"))
        self.iou = iou if iou is not None else float(os.getenv("IOU_THRESHOLD", "0.45"))
        self.timeout = timeout
        self.class_names = {}
        self.enabled_categories: set[str] | None = None
        self._session = requests.Session()

    def predict_frame(self, frame: np.ndarray) -> DetectionResult:
        h, w = frame.shape[:2]
        out = DetectionResult(width=w, height=h)

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            raise RuntimeError("Gagal encode frame ke JPEG")

        try:
            resp = self._session.post(
                f"{SERVERLESS_URL}/{self.model_id}",
                params={
                    "api_key": self.api_key,
                    # endpoint ini memakai skala persen
                    "confidence": int(self.detection_floor * 100),
                    "overlap": int(self.iou * 100),
                },
                data=base64.b64encode(buf.tobytes()),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=self.timeout,
            )
            resp.raise_for_status()
            payload = resp.json()
        except requests.RequestException as exc:
            # Jangan bikin loop webcam mati gara-gara satu frame gagal.
            print(f"[WARN] Request Roboflow gagal: {exc}")
            return self._finalize(out)

        for p in payload.get("predictions", []):
            category, is_violation = classify_label(p["class"])
            cx, cy, bw, bh = p["x"], p["y"], p["width"], p["height"]
            out.detections.append(
                Detection(
                    label=p["class"],
                    category=category,
                    confidence=round(float(p["confidence"]), 4),
                    bbox=[
                        int(cx - bw / 2),
                        int(cy - bh / 2),
                        int(cx + bw / 2),
                        int(cy + bh / 2),
                    ],
                    is_violation=is_violation,
                )
            )

        return self._finalize(out)
