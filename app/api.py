"""FastAPI service untuk PPE Detection."""
from __future__ import annotations

import base64
from functools import lru_cache
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from src.backends import build_detector, normalize
from src.detector import PPE_CLASSES, PPEDetector

app = FastAPI(
    title="PPE Detection API",
    description="YOLOv8 PPE detection service — 17 kelas model, "
                "8 kategori kepatuhan. Backend: PyTorch / OpenVINO / Roboflow.",
    version="1.2.0",
)

# Frontend `web/index.html` di-serve dari origin yang sama, jadi CORS sebenarnya
# tidak wajib. Tetap dibuka supaya file HTML-nya bisa dibuka langsung lewat
# file:// atau di-host terpisah (mis. GitHub Pages) sambil menunjuk API ini.
# Demo publik read-only tanpa auth — tidak ada yang bocor kalau origin bebas.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

WEB_DIR = Path(__file__).resolve().parents[1] / "web"


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
            # Ambang terendah yang dipakai server. Frontend memakainya sebagai
            # batas bawah slider confidence: menaikkan ambang bisa dilakukan
            # di client (tinggal saring), menurunkannya tidak — deteksi di
            # bawah angka ini sudah dibuang model sebelum sempat dikirim.
            "conf_threshold": det.detection_floor,
            "iou_threshold": det.iou,
            "ppe_categories": PPE_CLASSES,
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
async def predict(
    file: UploadFile = File(...),
    annotate: bool = Query(
        True,
        description="Sertakan gambar teranotasi sebagai base64 PNG. Set false "
                    "untuk mode realtime — client menggambar box sendiri dari "
                    "koordinat bbox, jadi tidak ada biaya encode + transfer PNG.",
    ),
) -> dict:
    """Deteksi PPE pada satu gambar. Kembalikan JSON (+ annotated image opsional)."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Hanya menerima file gambar")

    raw = await file.read()
    img = _decode_upload(raw)

    detector = get_detector()
    result = detector.predict_frame(img)
    payload = result.to_dict()

    if not annotate:
        return payload

    annotated = detector.render(img, result)
    ok, buf = cv2.imencode(".png", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Encode hasil gagal")

    payload["annotated_image_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return payload


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


# Mount paling akhir: route eksplisit di atas (/health, /predict, /docs) sudah
# terdaftar duluan sehingga tetap menang, dan sisanya jatuh ke frontend statis.
# `html=True` membuat `/` melayani index.html.
if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
