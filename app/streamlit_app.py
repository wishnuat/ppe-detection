"""Streamlit demo UI untuk PPE Detection.

Modes:
    - Upload gambar → deteksi + compliance status
    - Upload video → deteksi frame-by-frame
    - Webcam live (jika environment mendukung akses kamera)

UI memanggil detector lokal secara langsung (tidak lewat API) supaya
demo tetap jalan meski FastAPI service tidak dinyalakan.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# `streamlit run app/streamlit_app.py` menaruh folder app/ di sys.path, bukan
# root project — tanpa ini `import src.detector` gagal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends import build_detector
from src.detector import PPE_CLASSES, SELECTABLE_CATEGORIES, PPEDetector

st.set_page_config(
    page_title="PPE Detection Demo",
    page_icon="🦺",
    layout="wide",
)


@st.cache_resource
def load_detector(backend: str = "torch", device: str = "CPU") -> PPEDetector:
    """Cache per (backend, device) — compile OpenVINO makan beberapa detik,
    jadi jangan diulang tiap rerun Streamlit."""
    return build_detector(backend, device=device)


STATUS_STYLE = {
    "TERDETEKSI": ("✅", "#16a34a"),
    "PELANGGARAN": ("⚠️", "#dc2626"),
    "TIDAK TERDETEKSI": ("⚪", "#6b7280"),
}


def render_compliance(compliance: dict[str, str]) -> None:
    shown = [p for p in PPE_CLASSES if p in compliance]
    if not shown:
        st.info("Semua kategori APD dinonaktifkan di panel deteksi.")
        return
    cols = st.columns(len(shown))
    for col, ppe in zip(cols, shown):
        status = compliance.get(ppe, "TIDAK TERDETEKSI")
        icon, color = STATUS_STYLE[status]
        col.markdown(
            f"""
            <div style="border:1px solid #e5e7eb;border-radius:8px;
                        padding:12px;text-align:center;">
                <div style="font-size:22px;">{icon}</div>
                <div style="font-weight:600;text-transform:capitalize;">{ppe}</div>
                <div style="color:{color};font-size:13px;font-weight:600;">{status}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_detections_table(detections: list[dict]) -> None:
    if not detections:
        st.info("Tidak ada objek terdeteksi.")
        return
    st.dataframe(
        [
            {
                "Label": d["label"],
                "Confidence": f"{d['confidence']:.2%}",
                "BBox": d["bbox"],
                "Pelanggaran?": "Ya" if d["is_violation"] else "Tidak",
            }
            for d in detections
        ],
        use_container_width=True,
    )


CATEGORY_LABELS = {
    "helmet": "Helm",
    "glasses": "Kacamata",
    "mask": "Masker",
    "glove": "Sarung tangan",
    "shoes": "Sepatu",
    "vest": "Rompi",
    "ear_protection": "Pelindung telinga",
    "harness": "Harness",
    "person": "Orang (bukan APD)",
}


def category_panel() -> set[str]:
    """Panel aktif/nonaktif per kategori deteksi."""
    st.sidebar.markdown("### 🎚️ Kategori yang dideteksi")

    c1, c2 = st.sidebar.columns(2)
    if c1.button("Pilih semua", use_container_width=True):
        for cat in SELECTABLE_CATEGORIES:
            st.session_state[f"cat_{cat}"] = True
    if c2.button("Kosongkan", use_container_width=True):
        for cat in SELECTABLE_CATEGORIES:
            st.session_state[f"cat_{cat}"] = False

    enabled: set[str] = set()
    for cat in SELECTABLE_CATEGORIES:
        key = f"cat_{cat}"
        if key not in st.session_state:
            st.session_state[key] = True
        if st.sidebar.checkbox(CATEGORY_LABELS.get(cat, cat), key=key):
            enabled.add(cat)

    if not enabled:
        st.sidebar.warning("Tidak ada kategori aktif — tidak akan ada deteksi.")
    return enabled


BACKEND_CHOICES = {
    "⚡ OpenVINO INT8 (tercepat)": "openvino-int8",
    "🚀 OpenVINO FP32": "openvino",
    "🐍 PyTorch (referensi akurasi)": "torch",
    "☁️ Roboflow serverless (online)": "roboflow",
}


def sidebar() -> tuple[str, float, str, str]:
    st.sidebar.title("🦺 PPE Detection")
    st.sidebar.caption("YOLOv8 · OpenVINO · FastAPI · Streamlit")
    mode = st.sidebar.radio("Mode input", ["Gambar", "Video", "Webcam"])

    backend_label = st.sidebar.radio(
        "Backend inference",
        list(BACKEND_CHOICES),
        index=1,
        help="OpenVINO = IR terkompilasi, jauh lebih cepat di CPU/iGPU Intel "
             "(butuh `python scripts/export_openvino.py`). PyTorch = models/best.pt "
             "apa adanya. Roboflow = butuh internet.",
    )
    backend = BACKEND_CHOICES[backend_label]

    device = "CPU"
    if backend.startswith("openvino"):
        device = st.sidebar.selectbox(
            "Device OpenVINO",
            ["CPU", "GPU", "AUTO"],
            help="GPU = iGPU Intel (Iris Xe / UHD). AUTO = biarkan OpenVINO memilih.",
        )

    conf = st.sidebar.slider("Confidence threshold", 0.1, 0.9, 0.35, 0.05)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Kategori kepatuhan:** helmet, glasses, mask, glove, shoes, vest, "
        "ear_protection, harness — masing-masing punya label positif dan "
        "label pelanggaran di model."
    )
    return mode, conf, backend, device


def run_image(detector: PPEDetector) -> None:
    uploaded = st.file_uploader(
        "Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"]
    )
    if not uploaded:
        return
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Gambar tidak valid.")
        return
    result = detector.predict_frame(img)
    annotated = detector.render(img, result)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Deteksi", use_container_width=True)

    st.subheader("Compliance Status")
    render_compliance(result.compliance)
    st.subheader("Detail Deteksi")
    render_detections_table([d.to_dict() for d in result.detections])


def run_video(detector: PPEDetector) -> None:
    uploaded = st.file_uploader("Upload video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
    if not uploaded:
        return
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Memproses video ..."):
        out_path = detector.predict_video(tmp_path)

    st.success(f"Selesai. Output: `{out_path}`")
    st.video(out_path)


def run_webcam_snapshot(detector: PPEDetector) -> None:
    snap = st.camera_input("Ambil snapshot dari webcam")
    if snap is None:
        return
    data = np.frombuffer(snap.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    result = detector.predict_frame(img)
    annotated = detector.render(img, result)
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    st.subheader("Compliance Status")
    render_compliance(result.compliance)
    render_detections_table([d.to_dict() for d in result.detections])


def run_webcam_realtime(detector: PPEDetector) -> None:
    """Loop realtime: baca kamera di sisi server, stream frame teranotasi ke browser.

    Catatan: kamera yang dipakai adalah kamera **mesin yang menjalankan
    Streamlit**. Cocok untuk penggunaan lokal / edge box; kalau Streamlit
    di-deploy ke server remote, pakai mode Snapshot.
    """
    c1, c2, c3 = st.columns([1, 1, 2])
    camera_index = c1.number_input("Camera index", min_value=0, max_value=8, value=0, step=1)
    max_fps = c2.slider("Batas FPS", 1, 30, 10)

    if "cam_running" not in st.session_state:
        st.session_state.cam_running = False

    start, stop = c3.columns(2)
    if start.button("▶️ Mulai", use_container_width=True, type="primary"):
        st.session_state.cam_running = True
    if stop.button("⏹️ Stop", use_container_width=True):
        st.session_state.cam_running = False

    frame_slot = st.empty()
    status_slot = st.empty()
    table_slot = st.empty()

    if not st.session_state.cam_running:
        frame_slot.info("Tekan **Mulai** untuk menyalakan kamera.")
        return

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.session_state.cam_running = False
        st.error(f"Gagal membuka kamera index {camera_index}.")
        return

    min_dt = 1.0 / float(max_fps)
    fps = 0.0
    t_prev = time.perf_counter()
    try:
        while st.session_state.cam_running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Gagal membaca frame dari kamera.")
                break

            result = detector.predict_frame(frame)
            annotated = detector.render(frame, result)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
                caption=f"{fps:.1f} fps · backend {detector.model_path}",
            )

            violations = [p for p, s in result.compliance.items() if s == "PELANGGARAN"]
            with status_slot.container():
                if violations:
                    st.error("⚠️ PELANGGARAN: " + ", ".join(
                        CATEGORY_LABELS.get(v, v) for v in violations
                    ))
                else:
                    st.success("✅ Tidak ada pelanggaran terdeteksi")
                render_compliance(result.compliance)

            with table_slot.container():
                render_detections_table([d.to_dict() for d in result.detections])

            elapsed = time.perf_counter() - now
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
    finally:
        cap.release()


def run_webcam(detector: PPEDetector) -> None:
    sub_mode = st.radio(
        "Mode webcam",
        ["Realtime", "Snapshot"],
        horizontal=True,
        help="Realtime membaca kamera di mesin yang menjalankan Streamlit. "
             "Snapshot memakai kamera browser (aman untuk deploy remote).",
    )
    if sub_mode == "Realtime":
        run_webcam_realtime(detector)
    else:
        run_webcam_snapshot(detector)


def main() -> None:
    mode, conf, backend, device = sidebar()
    enabled = category_panel()

    try:
        detector = load_detector(backend, device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()
    detector.conf = conf
    detector.enabled_categories = enabled
    active_device = getattr(detector, "device", None)
    st.sidebar.caption(
        f"Model aktif: `{detector.model_path}`"
        + (f" · device `{active_device}`" if active_device else "")
    )

    st.title("PPE (Personal Protective Equipment) Detection")
    st.caption(
        "Deteksi kepatuhan penggunaan APD (helm, rompi, masker, kacamata, "
        "sepatu, sarung tangan) berbasis YOLOv8."
    )

    if mode == "Gambar":
        run_image(detector)
    elif mode == "Video":
        run_video(detector)
    else:
        run_webcam(detector)


if __name__ == "__main__":
    main()
