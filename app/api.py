"""FastAPI service untuk PPE Detection."""
from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from src.backends import build_detector, normalize
from src.detector import PPEDetector

app = FastAPI(
    title="PPE Detection API",
    description="YOLOv8 PPE detection service — 17 kelas model, "
                "8 kategori kepatuhan. Backend: PyTorch / OpenVINO / Roboflow.",
    version="1.1.0",
)


@lru_cache(maxsize=1)
def get_detector() -> PPEDetector:
    """Backend dipilih lewat env `INFERENCE_BACKEND` (default: torch).

    Di-cache supaya model hanya di-load/di-compile sekali per proses —
    compile OpenVINO makan waktu beberapa detik.
    """
    return build_detector()


@app.get("/health")
def health() -> dict:
    try:
        det = get_detector()
        return {
            "status": "ok",
            "backend": normalize(None),
            "device": getattr(det, "device", "cpu"),
            "model_path": det.model_path,
            "num_classes": len(det.class_names),
            "classes": list(det.class_names.values()),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "error": str(exc)}
        )


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar valid")
    return img


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Deteksi PPE pada satu gambar. Kembalikan JSON + annotated image (base64 PNG)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Hanya menerima file gambar")

    raw = await file.read()
    img = _decode_upload(raw)

    detector = get_detector()
    result = detector.predict_frame(img)
    annotated = detector.render(img, result)

    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode hasil gagal")

    return {
        **result.to_dict(),
        "annotated_image_b64": base64.b64encode(buf.tobytes()).decode("ascii"),
    }


@app.post("/predict/image")
async def predict_image_only(file: UploadFile = File(...)) -> StreamingResponse:
    """Sama dengan /predict tapi langsung return image (PNG) tanpa JSON."""
    raw = await file.read()
    img = _decode_upload(raw)
    detector = get_detector()
    result = detector.predict_frame(img)
    annotated = detector.render(img, result)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode hasil gagal")
    return StreamingResponse(BytesIO(buf.tobytes()), media_type="image/png")
