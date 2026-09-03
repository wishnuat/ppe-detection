"""Router FastAPI untuk fatigue detection & absensi.

Dipisah dari `app/api.py` supaya layanan PPE tetap bisa jalan sendiri: mesin
edge yang cuma butuh deteksi APD tidak perlu memuat empat model wajah, dan
kegagalan salah satu tidak menjatuhkan yang lain. `app/api.py` memasang router
ini secara opsional.

Satu hal yang perlu dipahami saat memakai endpoint HTTP untuk fatigue:
**kelelahan tidak bisa dinilai dari satu request.** PERCLOS, microsleep, dan
laju kedip semuanya butuh riwayat. Karena itu endpoint `/fatigue/analyze`
memakai state per-`session_id`: kirimkan frame berurutan dari kamera yang sama
dengan session_id yang sama, dan riwayatnya terakumulasi di server. Request
tanpa session_id diperlakukan sebagai satu frame lepas dan akan selalu
mengembalikan level TIDAK_DIKETAHUI — itu jujur, bukan bug.
"""
from __future__ import annotations

import base64
import time
from threading import Lock

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from src.fatigue.attendance import AttendanceBook
from src.fatigue.face import FaceDetector, build_embedder
from src.fatigue.pipeline import FatiguePipeline, PipelineConfig

router = APIRouter(prefix="/fatigue", tags=["fatigue"])

# Sesi menua sendiri: klien yang berhenti mengirim frame tidak selalu sempat
# memanggil endpoint reset, dan tanpa ini state-nya menumpuk sampai proses
# kehabisan memori.
SESSION_TTL = 600.0
MAX_SESSIONS = 32

_sessions: dict[str, tuple[FatiguePipeline, float]] = {}
_lock = Lock()
_shared_book: AttendanceBook | None = None


def _book() -> AttendanceBook:
    """Satu AttendanceBook dipakai bersama semua sesi.

    Kalau tiap sesi punya bukunya sendiri, cache embedding-nya terduplikasi dan
    pendaftaran karyawan baru tidak terlihat oleh sesi yang sudah berjalan.
    """
    global _shared_book
    if _shared_book is None:
        embedder = build_embedder()
        _shared_book = AttendanceBook(
            backend=embedder.backend, threshold=embedder.threshold
        )
    return _shared_book


def _evict_sessions(now: float) -> None:
    for key in [k for k, (_, seen) in _sessions.items() if now - seen > SESSION_TTL]:
        _sessions.pop(key)[0].close()
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k][1])
        _sessions.pop(oldest)[0].close()


def get_pipeline(session_id: str) -> FatiguePipeline:
    """Pipeline untuk satu sesi kamera, dibuat sekali lalu dipakai ulang."""
    now = time.time()
    with _lock:
        _evict_sessions(now)
        entry = _sessions.get(session_id)
        if entry is None:
            pipeline = FatiguePipeline(
                config=PipelineConfig(camera_name=session_id),
                attendance=_book(),
            )
            _sessions[session_id] = (pipeline, now)
            return pipeline
        _sessions[session_id] = (entry[0], now)
        return entry[0]


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="File bukan gambar valid")
    return img


# ---------------------------------------------------------------- status
@router.get("/health")
def health() -> dict:
    try:
        pipeline = get_pipeline("_health")
        return {
            "status": "ok",
            "components": pipeline.describe(),
            "attendance": _book().stats(),
            "active_sessions": len(_sessions),
        }
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(
            status_code=503, content={"status": "unavailable", "error": str(exc)}
        )


# ---------------------------------------------------------------- analisis
@router.post("/analyze")
async def analyze(
    file: UploadFile = File(...),
    session_id: str = Query(
        "default",
        description="Identitas aliran kamera. Frame berurutan dengan session_id "
                    "yang sama berbagi riwayat temporal — tanpa itu, level "
                    "fatigue tidak akan pernah keluar dari TIDAK_DIKETAHUI.",
    ),
    annotate: bool = Query(False, description="Sertakan gambar teranotasi (base64 PNG)."),
) -> dict:
    """Analisis satu frame dalam konteks sesi kamera."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Hanya menerima file gambar")

    img = _decode(await file.read())
    pipeline = get_pipeline(session_id)
    analysis = pipeline.process_frame(img)

    payload = analysis.to_dict()
    payload["session_id"] = session_id
    payload["checkins"] = [r.to_row() for r in pipeline.recent_checkins[-5:]]

    if annotate:
        annotated = pipeline.render(img, analysis)
        ok, buf = cv2.imencode(".png", annotated)
        if not ok:
            raise HTTPException(status_code=500, detail="Encode hasil gagal")
        payload["annotated_image_b64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return payload


@router.post("/session/{session_id}/reset")
def reset_session(session_id: str) -> dict:
    """Buang riwayat temporal satu sesi (ganti shift / ganti kamera)."""
    with _lock:
        entry = _sessions.pop(session_id, None)
    if entry is None:
        raise HTTPException(status_code=404, detail=f"Sesi '{session_id}' tidak ada")
    entry[0].close()
    return {"status": "reset", "session_id": session_id}


@router.get("/sessions")
def list_sessions() -> dict:
    now = time.time()
    return {
        "sessions": [
            {"session_id": k, "idle_seconds": round(now - seen, 1),
             "people": pipeline.num_tracked}
            for k, (pipeline, seen) in sorted(_sessions.items())
        ],
        "ttl_seconds": SESSION_TTL,
    }


# ---------------------------------------------------------------- absensi
@router.get("/employees")
def list_employees(active_only: bool = Query(False)) -> dict:
    return {"employees": [e.to_dict() for e in _book().list_employees(active_only)]}


@router.post("/employees")
async def enroll(
    employee_id: str = Form(...),
    name: str = Form(...),
    department: str = Form(""),
    files: list[UploadFile] = File(...),
) -> dict:
    """Daftarkan karyawan dengan satu atau beberapa foto wajah.

    Foto yang wajahnya tidak terdeteksi ditolak dan disebutkan satu per satu,
    bukan diam-diam dilewati: pendaftaran yang tampak berhasil padahal tidak
    menyimpan apa pun akan muncul berbulan-bulan kemudian sebagai "kok orang
    ini tidak pernah terbaca".
    """
    book = _book()
    detector = FaceDetector(min_face=40)
    embedder = build_embedder()
    book.add_employee(employee_id, name, department)

    accepted, rejected = 0, []
    for upload in files:
        img = _decode(await upload.read())
        faces = detector.detect(img)
        if not faces:
            rejected.append({"file": upload.filename, "reason": "wajah tidak terdeteksi"})
            continue
        if len(faces) > 1:
            rejected.append({"file": upload.filename,
                             "reason": f"ada {len(faces)} wajah, harus satu orang"})
            continue
        book.add_embedding(employee_id, embedder.embed(img, faces[0]),
                           source=upload.filename or "")
        accepted += 1

    if accepted == 0:
        raise HTTPException(
            status_code=400,
            detail={"message": "Tidak ada foto yang bisa dipakai", "rejected": rejected},
        )
    return {
        "employee_id": employee_id, "name": name,
        "accepted": accepted, "rejected": rejected,
        "total_embeddings": next(
            (e.num_embeddings for e in book.list_employees()
             if e.employee_id == employee_id), 0
        ),
    }


@router.delete("/employees/{employee_id}")
def delete_employee(employee_id: str) -> dict:
    """Hapus karyawan berikut seluruh data biometrik & log-nya. Permanen."""
    if not _book().delete_employee(employee_id):
        raise HTTPException(status_code=404, detail=f"'{employee_id}' tidak ditemukan")
    return {"status": "deleted", "employee_id": employee_id}


@router.get("/attendance")
def attendance_log(
    employee_id: str | None = Query(None),
    since: float | None = Query(None, description="Unix timestamp batas bawah."),
    today_only: bool = Query(False),
    limit: int = Query(200, le=2000),
) -> dict:
    book = _book()
    records = book.today() if today_only else book.records(
        since=since, employee_id=employee_id, limit=limit
    )
    return {"count": len(records), "records": [r.to_row() for r in records]}
