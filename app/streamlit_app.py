"""Streamlit demo UI untuk PPE Detection — versi lokal/operator.

Mode input:
    - Gambar  → deteksi + compliance + unduh hasil (JPG/JSON)
    - Video   → proses frame-by-frame dengan progress, event log, unduh mp4
    - Webcam  → realtime (kamera mesin ini) atau snapshot (kamera browser)

Yang bisa diatur user (semua di sidebar):
    - Sensitivitas: preset, confidence & IoU global, plus ambang per kategori
      untuk kelas yang modelnya lemah (mis. `glove`).
    - Kategori yang dideteksi: matikan APD yang tidak relevan di area kerja.
    - Alert: kategori mana yang boleh membunyikan alarm, berapa frame
      pelanggaran harus bertahan, cooldown, suara, dan snapshot bukti.
    - Profil: simpan/muat kombinasi setting di atas ke config/ui_profiles.json.

UI memanggil detector lokal langsung (tidak lewat FastAPI) supaya demo tetap
jalan meski service API tidak dinyalakan.
"""
from __future__ import annotations

import json
import math
import struct
import sys
import tempfile
import time
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# `streamlit run app/streamlit_app.py` menaruh folder app/ di sys.path, bukan
# root project — tanpa ini `import src.detector` gagal.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alerts import AlertEngine, SessionStats, events_to_csv
from src.backends import build_detector
from src.detector import (
    PPE_CLASSES,
    RAW_LABEL_MAP,
    SELECTABLE_CATEGORIES,
    PPEDetector,
)

st.set_page_config(page_title="PPE Detection Demo", page_icon="🦺", layout="wide")

OUTPUT_DIR = PROJECT_ROOT / "outputs"
SNAPSHOT_DIR = OUTPUT_DIR / "violations"
SESSION_DIR = OUTPUT_DIR / "sessions"
PROFILE_PATH = PROJECT_ROOT / "config" / "ui_profiles.json"

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

STATUS_STYLE = {
    "TERDETEKSI": ("✅", "#16a34a"),
    "PELANGGARAN": ("⚠️", "#dc2626"),
    "TIDAK TERDETEKSI": ("⚪", "#6b7280"),
}

BACKEND_CHOICES = {
    "⚡ OpenVINO INT8 (tercepat)": "openvino-int8",
    "🚀 OpenVINO FP32": "openvino",
    "🐍 PyTorch (referensi akurasi)": "torch",
    "☁️ Roboflow serverless (online)": "roboflow",
}

# Preset sensitivitas: (confidence, iou).
# Longgar = tangkap sebanyak mungkin, konsekuensinya false alarm naik.
# Ketat   = hanya deteksi yang model-nya yakin, cocok untuk logging otomatis.
SENSITIVITY_PRESETS = {
    "🟢 Longgar — jangan sampai kelewat": (0.20, 0.50),
    "🟡 Seimbang (default)": (0.35, 0.45),
    "🔴 Ketat — minim false alarm": (0.55, 0.40),
}
CUSTOM_PRESET = "⚙️ Custom"

# Kategori yang punya label pelanggaran di dataset. `vest` & `harness` tidak
# punya lawan negatifnya, jadi tidak akan pernah memicu alert.
VIOLATABLE = sorted(
    {cat for cat, is_violation in RAW_LABEL_MAP.values() if is_violation}
)


# --------------------------------------------------------------------------
# Resource & state
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model ...")
def load_detector(backend: str, device: str) -> PPEDetector:
    """Cache per (backend, device) — compile OpenVINO makan beberapa detik,
    jadi jangan diulang tiap rerun Streamlit."""
    return build_detector(backend, device=device)


BEEP_RATE = 22050


@st.cache_data
def beep_pcm() -> bytes:
    """Sampel nada alarm (PCM 16-bit mono), dibangkitkan sendiri — tanpa file aset."""
    duration, freq = 0.28, 880.0
    total = int(BEEP_RATE * duration)
    frames = bytearray()
    for i in range(total):
        # Fade-out supaya tidak ada 'klik' di ujung nada.
        envelope = 1.0 - (i / total)
        value = int(22000 * envelope * math.sin(2 * math.pi * freq * i / BEEP_RATE))
        frames += struct.pack("<h", value)
    return bytes(frames)


def beep_wav(variant: int = 0) -> bytes:
    """Bungkus nada alarm jadi WAV.

    `variant` menambahkan ekor senyap beberapa milidetik. Kelihatan aneh, tapi
    perlu: Streamlit memberi ID otomatis pada elemen media berdasarkan isinya,
    jadi dua `st.audio` dengan byte identik dalam satu script run dianggap
    duplikat dan melempar StreamlitDuplicateElementId — dan loop realtime
    memang membunyikan alarm berkali-kali dalam satu run. Versi Streamlit di
    requirements belum menerima argumen `key` pada `st.audio`, jadi keunikan
    dibuat dari datanya sendiri.
    """
    pcm = beep_pcm() + b"\x00\x00" * (variant % 4096)
    buf = BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(BEEP_RATE)
        wav.writeframes(pcm)
    return buf.getvalue()


def init_state() -> None:
    st.session_state.setdefault("alert_engine", AlertEngine())
    st.session_state.setdefault("stats", SessionStats())
    st.session_state.setdefault("cam_running", False)
    st.session_state.setdefault("last_recording", None)
    st.session_state.setdefault("preset", "🟡 Seimbang (default)")
    st.session_state.setdefault("conf", 0.35)
    st.session_state.setdefault("iou", 0.45)
    st.session_state.setdefault("per_category_conf", False)

    # Nilai dari profil yang baru dimuat: harus ditulis SEBELUM widget dibuat,
    # karena session_state milik widget tidak boleh diubah setelah widget-nya
    # ter-render di run yang sama.
    pending = st.session_state.pop("_pending_settings", None)
    if pending:
        for key, value in pending.items():
            st.session_state[key] = value


def apply_preset() -> None:
    """Callback selectbox preset — jalan sebelum widget lain di-render."""
    preset = st.session_state.preset
    if preset in SENSITIVITY_PRESETS:
        conf, iou = SENSITIVITY_PRESETS[preset]
        st.session_state.conf = conf
        st.session_state.iou = iou


def mark_custom() -> None:
    """Begitu slider digeser manual, preset tidak lagi mewakili nilainya."""
    st.session_state.preset = CUSTOM_PRESET


# --------------------------------------------------------------------------
# Profil setting
# --------------------------------------------------------------------------
PROFILE_KEYS = (
    ["conf", "iou", "per_category_conf"]
    + [f"cat_{c}" for c in SELECTABLE_CATEGORIES]
    + [f"catconf_{c}" for c in SELECTABLE_CATEGORIES]
    + [f"alert_{c}" for c in VIOLATABLE]
    + ["alert_min_frames", "alert_cooldown", "alert_require_person",
       "alert_sound", "alert_snapshot"]
)


def load_profiles() -> dict[str, dict]:
    if not PROFILE_PATH.exists():
        return {}
    try:
        return json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def save_profile(name: str) -> None:
    profiles = load_profiles()
    profiles[name] = {
        key: st.session_state[key] for key in PROFILE_KEYS if key in st.session_state
    }
    PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROFILE_PATH.write_text(
        json.dumps(profiles, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def sidebar_model() -> tuple[str, str, str]:
    st.sidebar.title("🦺 PPE Detection")
    st.sidebar.caption("YOLOv8 · OpenVINO · FastAPI · Streamlit")
    mode = st.sidebar.radio("Mode input", ["Gambar", "Video", "Webcam"], horizontal=True)

    with st.sidebar.expander("🧠 Model & backend", expanded=False):
        backend_label = st.radio(
            "Backend inference",
            list(BACKEND_CHOICES),
            index=1,
            help="OpenVINO = IR terkompilasi, jauh lebih cepat di CPU/iGPU Intel "
                 "(butuh `python scripts/export_openvino.py`). PyTorch = "
                 "models/best.pt apa adanya. Roboflow = butuh internet.",
        )
        backend = BACKEND_CHOICES[backend_label]
        device = "CPU"
        if backend.startswith("openvino"):
            device = st.selectbox(
                "Device OpenVINO", ["CPU", "GPU", "AUTO"],
                help="GPU = iGPU Intel (Iris Xe / UHD). AUTO = biar OpenVINO memilih.",
            )
    return mode, backend, device


def sidebar_sensitivity() -> tuple[float, float, dict[str, float]]:
    with st.sidebar.expander("🎚️ Sensitivitas", expanded=True):
        st.selectbox(
            "Preset", [*SENSITIVITY_PRESETS, CUSTOM_PRESET],
            key="preset", on_change=apply_preset,
            help="Preset hanya mengisi dua slider di bawah. Geser slider "
                 "kapan saja — preset otomatis jadi Custom.",
        )
        conf = st.slider(
            "Confidence threshold", 0.05, 0.95, step=0.05,
            key="conf", on_change=mark_custom,
            help="Makin rendah = makin sensitif (lebih banyak deteksi, lebih "
                 "banyak false alarm).",
        )
        iou = st.slider(
            "IoU / NMS threshold", 0.10, 0.90, step=0.05,
            key="iou", on_change=mark_custom,
            help="Seberapa besar dua box boleh bertumpuk sebelum salah satunya "
                 "dibuang. Turunkan kalau box dobel, naikkan kalau objek "
                 "berdempetan malah hilang.",
        )

        st.checkbox(
            "Ambang khusus per kategori", key="per_category_conf",
            help="Model ini lemah di kelas kecil/ambigu (hand_noglove, "
                 "No_Glasses). Turunkan ambangnya sendiri tanpa membuat "
                 "seluruh deteksi jadi berisik.",
        )

        overrides: dict[str, float] = {}
        if st.session_state.per_category_conf:
            if st.button("Samakan semua ke global", use_container_width=True):
                for cat in SELECTABLE_CATEGORIES:
                    st.session_state[f"catconf_{cat}"] = conf
                st.rerun()
            for cat in SELECTABLE_CATEGORIES:
                key = f"catconf_{cat}"
                st.session_state.setdefault(key, conf)
                value = st.slider(
                    CATEGORY_LABELS.get(cat, cat), 0.05, 0.95, step=0.05, key=key
                )
                if abs(value - conf) > 1e-9:
                    overrides[cat] = value
        return conf, iou, overrides


def sidebar_categories() -> set[str]:
    with st.sidebar.expander("👁️ Kategori yang dideteksi", expanded=True):
        c1, c2 = st.columns(2)
        if c1.button("Pilih semua", use_container_width=True):
            for cat in SELECTABLE_CATEGORIES:
                st.session_state[f"cat_{cat}"] = True
            st.rerun()
        if c2.button("Kosongkan", use_container_width=True):
            for cat in SELECTABLE_CATEGORIES:
                st.session_state[f"cat_{cat}"] = False
            st.rerun()

        enabled: set[str] = set()
        for cat in SELECTABLE_CATEGORIES:
            key = f"cat_{cat}"
            st.session_state.setdefault(key, True)
            if st.checkbox(CATEGORY_LABELS.get(cat, cat), key=key):
                enabled.add(cat)

    if not enabled:
        st.sidebar.warning("Tidak ada kategori aktif — tidak akan ada deteksi.")
    return enabled


def sidebar_alerts(enabled: set[str]) -> dict:
    """Panel kebijakan alert. Terpisah dari panel deteksi: sebuah kategori bisa
    tetap digambar & dicatat tanpa harus membunyikan alarm."""
    with st.sidebar.expander("🚨 Alert", expanded=True):
        st.caption(
            "Kategori yang dicentang di sini yang memicu alarm. "
            "Rompi & harness tidak punya label pelanggaran di dataset, "
            "jadi tidak pernah bisa dialarmkan."
        )
        c1, c2 = st.columns(2)
        if c1.button("Semua alert", use_container_width=True):
            for cat in VIOLATABLE:
                st.session_state[f"alert_{cat}"] = True
            st.rerun()
        if c2.button("Bisukan", use_container_width=True):
            for cat in VIOLATABLE:
                st.session_state[f"alert_{cat}"] = False
            st.rerun()

        alert_categories: set[str] = set()
        for cat in VIOLATABLE:
            key = f"alert_{cat}"
            st.session_state.setdefault(key, True)
            off = cat not in enabled
            checked = st.checkbox(
                CATEGORY_LABELS.get(cat, cat), key=key, disabled=off,
                help="Kategori ini sedang dimatikan di panel deteksi." if off else None,
            )
            if checked and not off:
                alert_categories.add(cat)

        st.markdown("---")
        # Semua widget di bawah pakai setdefault + tanpa `value=`: nilai awalnya
        # datang dari session_state, supaya "Muat profil" tidak bentrok dengan
        # default hardcoded.
        st.session_state.setdefault("alert_min_frames", 3)
        st.session_state.setdefault("alert_cooldown", 15)
        st.session_state.setdefault("alert_require_person", False)
        st.session_state.setdefault("alert_sound", True)
        st.session_state.setdefault("alert_snapshot", True)

        min_frames = st.slider(
            "Pelanggaran harus bertahan (frame)", 1, 15, key="alert_min_frames",
            help="Peredam kedipan: alarm baru bunyi kalau pelanggaran terlihat "
                 "di N frame berturut-turut.",
        )
        cooldown = st.slider(
            "Cooldown per kategori (detik)", 0, 120, step=5, key="alert_cooldown",
            help="Jeda minimum sebelum kategori yang sama boleh dialarmkan lagi.",
        )
        require_person = st.checkbox(
            "Hanya alarm kalau ada orang di frame", key="alert_require_person",
            help="Meredam alarm dari helm/rompi tergeletak. Butuh kategori "
                 "'Orang' aktif di panel deteksi.",
        )
        sound = st.checkbox("Bunyikan suara saat alarm", key="alert_sound")
        snapshot = st.checkbox(
            "Simpan snapshot bukti", key="alert_snapshot",
            help=f"Frame teranotasi disimpan ke {SNAPSHOT_DIR.relative_to(PROJECT_ROOT)}/",
        )

    if require_person and "person" not in enabled:
        st.sidebar.warning(
            "Opsi 'hanya alarm kalau ada orang' aktif tapi kategori **Orang** "
            "dimatikan — alarm tidak akan pernah bunyi."
        )

    return {
        "categories": alert_categories,
        "min_frames": min_frames,
        "cooldown": float(cooldown),
        "require_person": require_person,
        "sound": sound,
        "snapshot": snapshot,
    }


def sidebar_profiles() -> None:
    with st.sidebar.expander("💾 Profil setting", expanded=False):
        profiles = load_profiles()
        name = st.text_input("Nama profil", placeholder="mis. Gudang malam")
        if st.button("Simpan profil", use_container_width=True, disabled=not name.strip()):
            save_profile(name.strip())
            st.success(f"Profil `{name.strip()}` tersimpan.")

        if profiles:
            chosen = st.selectbox("Muat profil", sorted(profiles))
            if st.button("Terapkan", use_container_width=True):
                # Ditunda ke run berikutnya lewat init_state(): widget yang
                # sudah ter-render tidak boleh diubah nilainya sekarang.
                st.session_state["_pending_settings"] = profiles[chosen]
                st.rerun()
        else:
            st.caption("Belum ada profil tersimpan.")


# --------------------------------------------------------------------------
# Komponen tampilan
# --------------------------------------------------------------------------
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
                <div style="font-weight:600;">{CATEGORY_LABELS.get(ppe, ppe)}</div>
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
                "Kategori": CATEGORY_LABELS.get(d["category"], d["category"]),
                "Confidence": f"{d['confidence']:.2%}",
                "BBox": d["bbox"],
                "Pelanggaran?": "Ya" if d["is_violation"] else "Tidak",
            }
            for d in detections
        ],
        use_container_width=True,
        hide_index=True,
    )


def render_metrics(stats: SessionStats, engine: AlertEngine, fps: float) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("FPS", f"{fps:.1f}")
    c2.metric("Frame diproses", stats.frames)
    c3.metric(
        "Kepatuhan", f"{stats.compliance_rate:.0f}%",
        help="Persentase frame yang bebas pelanggaran sepanjang sesi.",
    )
    c4.metric("Alarm", len(engine.events))


def render_event_log(
    engine: AlertEngine, key: str, video_fps: float | None = None
) -> None:
    """Log alarm. Untuk mode video, kolom waktu memakai posisi di dalam video
    (mm:ss) — jam dinding saat memproses file tidak ada artinya bagi operator."""
    st.subheader("📋 Log pelanggaran")
    if not engine.events:
        st.info("Belum ada alarm pada sesi ini.")
        return

    def when(event) -> str:
        if video_fps:
            secs = event.frame_index / video_fps
            return f"{int(secs // 60):02d}:{int(secs % 60):02d}"
        return event.clock

    rows = [
        {
            "Waktu": when(e),
            "Kategori": CATEGORY_LABELS.get(e.category, e.category),
            "Confidence": f"{e.confidence:.2%}",
            "Frame": e.frame_index,
            "Bukti": Path(e.snapshot).name if e.snapshot else "-",
        }
        for e in reversed(engine.events[-100:])
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    c1, c2 = st.columns([1, 3])
    c1.download_button(
        "⬇️ Unduh CSV",
        data=events_to_csv(engine.events).encode("utf-8"),
        file_name=f"ppe_events_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv",
        use_container_width=True,
        key=f"csv_{key}",
    )
    if c2.button("🧹 Reset log & statistik", key=f"reset_{key}"):
        engine.reset()
        st.session_state.stats.reset()
        st.rerun()


def render_snapshots(engine: AlertEngine) -> None:
    shots = [e.snapshot for e in engine.events if e.snapshot]
    if not shots:
        return
    st.subheader("📸 Bukti terakhir")
    for col, path in zip(st.columns(4), reversed(shots[-4:])):
        if Path(path).exists():
            col.image(path, caption=Path(path).name, use_container_width=True)


def save_snapshot(frame: np.ndarray, category: str) -> str:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = SNAPSHOT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{category}.jpg"
    cv2.imwrite(str(path), frame)
    return str(path)


# Hasil percobaan codec H.264, di-cache: OpenCV Windows sering tidak membawa
# openh264.dll dan gagal berisik ke stderr — cukup sekali per proses.
_H264_AVAILABLE: bool | None = None


def open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 kalau tersedia (bisa diputar langsung di browser), fallback mp4v."""
    global _H264_AVAILABLE
    path.parent.mkdir(parents=True, exist_ok=True)
    fps = max(1.0, fps)

    if _H264_AVAILABLE is not False:
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"avc1"), fps, size)
        _H264_AVAILABLE = writer.isOpened()
        if _H264_AVAILABLE:
            return writer
        writer.release()

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"Gagal membuat file video di {path}")
    return writer


def play_beep(slot) -> None:
    """Bunyikan alarm sekali. Kalau gagal, sesi monitoring tetap jalan —
    bunyi itu pelengkap, bukan alasan mematikan kamera di tengah shift."""
    seq = st.session_state.get("beep_seq", 0) + 1
    st.session_state["beep_seq"] = seq
    try:
        with slot.container():
            st.audio(beep_wav(seq), format="audio/wav", autoplay=True)
    except Exception as exc:  # noqa: BLE001 - lihat docstring
        print(f"[WARN] Gagal membunyikan alarm: {exc}")


def handle_alerts(
    engine: AlertEngine, result, annotated: np.ndarray, cfg: dict,
    frame_index: int, beep_slot=None,
) -> None:
    """Update alert engine + efek sampingnya (snapshot bukti, bunyi)."""
    fired = engine.update(result, frame_index=frame_index)
    if not fired:
        return
    if cfg["snapshot"]:
        for event in fired:
            event.snapshot = save_snapshot(annotated, event.category)
    if cfg["sound"] and beep_slot is not None:
        play_beep(beep_slot)


# --------------------------------------------------------------------------
# Mode: gambar
# --------------------------------------------------------------------------
def run_image(detector: PPEDetector, cfg: dict) -> None:
    uploaded = st.file_uploader("Upload gambar (jpg/png)", type=["jpg", "jpeg", "png"])
    if not uploaded:
        st.info("Pilih gambar untuk mulai. Semua setting di sidebar langsung berlaku.")
        return

    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Gambar tidak valid.")
        return

    result = detector.predict_frame(img)
    annotated = detector.render(img, result)

    # Gambar tunggal: satu frame = satu kesempatan, jadi debounce dimatikan.
    single = AlertEngine(
        categories=cfg["categories"], min_frames=1, cooldown=0.0,
        require_person=cfg["require_person"],
    )
    fired = single.update(result)

    c1, c2 = st.columns(2)
    c1.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original",
             use_container_width=True)
    c2.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Deteksi",
             use_container_width=True)

    if fired:
        st.error("⚠️ PELANGGARAN: " + ", ".join(
            CATEGORY_LABELS.get(e.category, e.category) for e in fired
        ))
    else:
        st.success("✅ Tidak ada pelanggaran pada kategori yang dialarmkan.")

    st.subheader("Compliance Status")
    render_compliance(result.compliance)
    st.subheader("Detail Deteksi")
    render_detections_table([d.to_dict() for d in result.detections])

    ok, buf = cv2.imencode(".jpg", annotated)
    stem = Path(uploaded.name).stem
    c1, c2 = st.columns(2)
    if ok:
        c1.download_button(
            "⬇️ Unduh gambar teranotasi", data=buf.tobytes(),
            file_name=f"annotated_{stem}.jpg", mime="image/jpeg",
            use_container_width=True,
        )
    c2.download_button(
        "⬇️ Unduh hasil JSON",
        data=json.dumps(result.to_dict(), indent=2, ensure_ascii=False).encode("utf-8"),
        file_name=f"result_{stem}.json", mime="application/json",
        use_container_width=True,
    )


# --------------------------------------------------------------------------
# Mode: video
# --------------------------------------------------------------------------
def run_video(detector: PPEDetector, engine: AlertEngine, cfg: dict) -> None:
    uploaded = st.file_uploader("Upload video (mp4/mov/avi)", type=["mp4", "mov", "avi"])
    if not uploaded:
        st.info("Pilih video untuk diproses. Hasilnya bisa diunduh beserta log CSV.")
        return

    stride = st.slider(
        "Proses tiap N frame", 1, 10, 1,
        help="Naikkan untuk mempercepat video panjang: frame yang dilewati "
             "tetap ditulis ke output, tapi tanpa deteksi baru.",
    )
    if not st.button("▶️ Proses video", type="primary"):
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Gagal membuka video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    out_path = OUTPUT_DIR / f"annotated_{Path(uploaded.name).stem}.mp4"

    # Video punya timeline sendiri: cooldown dihitung dalam detik video,
    # bukan jam dinding, supaya log-nya bisa dirujuk ke menit ke berapa.
    engine.reset()
    st.session_state.stats.reset()
    stats = st.session_state.stats

    writer = open_writer(out_path, fps, size)
    progress = st.progress(0.0, text="Memproses ...")
    preview = st.empty()
    index = 0
    result = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0 or result is None:
                result = detector.predict_frame(frame)
                stats.update(result)
                # `now` memakai jam video, bukan jam dinding: cooldown 15 detik
                # harus berarti 15 detik rekaman, bukan 15 detik pemrosesan.
                fired = engine.update(result, now=index / fps, frame_index=index)
                annotated = detector.render(frame, result)
                for event in fired:
                    event.timestamp = time.time()
                    if cfg["snapshot"]:
                        event.snapshot = save_snapshot(annotated, event.category)
            else:
                annotated = detector.render(frame, result)

            writer.write(annotated)
            index += 1
            if total and index % 10 == 0:
                progress.progress(
                    min(1.0, index / total), text=f"Frame {index}/{total}"
                )
                preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                              use_container_width=True)
    finally:
        cap.release()
        writer.release()

    progress.progress(1.0, text=f"Selesai — {index} frame")
    preview.empty()
    st.success(f"Output: `{out_path}`")
    if out_path.exists():
        data = out_path.read_bytes()
        st.video(data)
        st.download_button(
            "⬇️ Unduh video hasil", data=data,
            file_name=out_path.name, mime="video/mp4",
        )
        if not _H264_AVAILABLE:
            st.caption(
                "Codec H.264 tidak tersedia di OpenCV ini, output ditulis "
                "dengan mp4v — beberapa browser tidak bisa memutarnya. "
                "Unduh filenya kalau preview di atas kosong."
            )

    proc_fps = index / stats.duration if stats.duration else 0.0
    render_metrics(stats, engine, fps=proc_fps)
    render_event_log(engine, key="video", video_fps=fps)
    render_snapshots(engine)


# --------------------------------------------------------------------------
# Mode: webcam
# --------------------------------------------------------------------------
def run_webcam_snapshot(detector: PPEDetector, cfg: dict) -> None:
    snap = st.camera_input("Ambil snapshot dari webcam")
    if snap is None:
        return
    data = np.frombuffer(snap.getvalue(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    result = detector.predict_frame(img)
    annotated = detector.render(img, result)

    single = AlertEngine(categories=cfg["categories"], min_frames=1, cooldown=0.0,
                         require_person=cfg["require_person"])
    fired = single.update(result)

    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
    if fired:
        st.error("⚠️ PELANGGARAN: " + ", ".join(
            CATEGORY_LABELS.get(e.category, e.category) for e in fired))
    st.subheader("Compliance Status")
    render_compliance(result.compliance)
    render_detections_table([d.to_dict() for d in result.detections])


def run_webcam_realtime(detector: PPEDetector, engine: AlertEngine, cfg: dict) -> None:
    """Loop realtime: baca kamera di sisi server, stream frame teranotasi ke browser.

    Kamera yang dipakai adalah kamera **mesin yang menjalankan Streamlit**.
    Cocok untuk pemakaian lokal / edge box; kalau Streamlit di-deploy ke server
    remote, pakai mode Snapshot.
    """
    c1, c2, c3 = st.columns([1, 1, 1])
    camera_index = c1.number_input("Camera index", 0, 8, 0, 1)
    max_fps = c2.slider("Batas FPS", 1, 30, 10)
    mirror = c3.checkbox("Cermin (mirror)", value=True,
                         help="Balik horizontal — lebih natural untuk kamera menghadap operator.")
    record = c3.checkbox("⏺️ Rekam sesi ke mp4", value=False)

    start, stop = st.columns(2)
    if start.button("▶️ Mulai", use_container_width=True, type="primary"):
        st.session_state.cam_running = True
        st.session_state.stats.reset()
        engine.reset()
    if stop.button("⏹️ Stop", use_container_width=True):
        st.session_state.cam_running = False

    beep_slot = st.empty()
    status_slot = st.empty()
    metric_slot = st.empty()
    frame_slot = st.empty()
    table_slot = st.empty()

    if not st.session_state.cam_running:
        frame_slot.info("Tekan **Mulai** untuk menyalakan kamera.")
        last = st.session_state.last_recording
        if last and Path(last).exists():
            st.success(f"Rekaman tersimpan: `{last}`")
            st.download_button(
                "⬇️ Unduh rekaman", data=Path(last).read_bytes(),
                file_name=Path(last).name, mime="video/mp4",
            )
            if not _H264_AVAILABLE:
                st.caption(
                    "Ditulis dengan codec mp4v (H.264 tidak tersedia di OpenCV "
                    "ini) — putar dengan VLC/Windows Media Player kalau browser "
                    "menolak."
                )
        render_metrics(st.session_state.stats, engine, fps=0.0)
        render_event_log(engine, key="cam")
        render_snapshots(engine)
        return

    # CAP_DSHOW = backend DirectShow Windows; jauh lebih cepat dibuka daripada
    # MSMF default. Di OS lain fallback ke backend bawaan.
    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.session_state.cam_running = False
        st.error(f"Gagal membuka kamera index {camera_index}.")
        return

    writer = None
    if record:
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480)
        # FPS rekaman diukur dulu, tidak diasumsikan sama dengan "Batas FPS":
        # kalau ditulis 10 fps padahal inference cuma sanggup 4 fps, video
        # hasilnya jadi 2,5x lebih cepat dari kejadian aslinya.
        with st.spinner("Mengukur kecepatan inference untuk FPS rekaman ..."):
            t0 = time.perf_counter()
            probes = 0
            for _ in range(8):
                ok, probe = cap.read()
                if not ok:
                    break
                detector.predict_frame(probe)
                probes += 1
        measured = probes / max(time.perf_counter() - t0, 1e-6) if probes else float(max_fps)
        rec_fps = max(1.0, min(float(max_fps), measured))
        path = SESSION_DIR / f"session_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = open_writer(path, rec_fps, size)
        st.session_state.last_recording = str(path)
        st.toast(f"⏺️ Merekam ke {path.name} @ {rec_fps:.1f} fps")

    stats = st.session_state.stats
    min_dt = 1.0 / float(max_fps)
    fps = 0.0
    index = 0
    t_prev = time.perf_counter()
    try:
        while st.session_state.cam_running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Gagal membaca frame dari kamera.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            result = detector.predict_frame(frame)
            annotated = detector.render(frame, result)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            stats.update(result)
            handle_alerts(engine, result, annotated, cfg, index, beep_slot)
            if writer is not None:
                writer.write(annotated)
            index += 1

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB",
                use_container_width=True,
                caption=f"{fps:.1f} fps · {Path(detector.model_path).name}"
                        + (" · ⏺️ merekam" if writer is not None else ""),
            )
            with metric_slot.container():
                render_metrics(stats, engine, fps)
            with status_slot.container():
                if engine.active:
                    st.error("⚠️ PELANGGARAN: " + ", ".join(
                        CATEGORY_LABELS.get(c, c) for c in sorted(engine.active)))
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
        if writer is not None:
            writer.release()


def run_webcam(detector: PPEDetector, engine: AlertEngine, cfg: dict) -> None:
    sub_mode = st.radio(
        "Mode webcam", ["Realtime", "Snapshot"], horizontal=True,
        help="Realtime membaca kamera di mesin yang menjalankan Streamlit. "
             "Snapshot memakai kamera browser (aman untuk deploy remote).",
    )
    if sub_mode == "Realtime":
        run_webcam_realtime(detector, engine, cfg)
    else:
        run_webcam_snapshot(detector, cfg)


# --------------------------------------------------------------------------
def main() -> None:
    init_state()

    # Pemilih modul ditaruh paling atas di sidebar: sisa sidebar-nya berbeda
    # total antara APD dan fatigue, dan menampilkan keduanya sekaligus akan
    # membuat operator mengatur hal yang tidak sedang ia pakai.
    module = st.sidebar.segmented_control(
        "Modul", ["APD", "Fatigue & absensi"], default="APD",
        key="active_module", label_visibility="collapsed",
    ) or "APD"

    if module == "Fatigue & absensi":
        # Import ditunda sampai modulnya dipilih — mediapipe, torch, dan tiga
        # model wajah butuh beberapa detik dan ratusan MB, dan pengguna yang
        # hanya memantau APD tidak perlu membayarnya.
        from app import fatigue_ui

        fatigue_ui.render()
        return

    mode, backend, device = sidebar_model()
    conf, iou, overrides = sidebar_sensitivity()
    enabled = sidebar_categories()
    cfg = sidebar_alerts(enabled)
    sidebar_profiles()

    try:
        detector = load_detector(backend, device)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        st.error(str(exc))
        st.stop()

    detector.conf = conf
    detector.iou = iou
    detector.category_conf = overrides or None
    detector.enabled_categories = enabled

    engine: AlertEngine = st.session_state.alert_engine
    engine.configure(
        categories=cfg["categories"],
        min_frames=cfg["min_frames"],
        cooldown=cfg["cooldown"],
        require_person=cfg["require_person"],
    )

    active_device = getattr(detector, "device", None)
    st.sidebar.caption(
        f"Model aktif: `{detector.model_path}`"
        + (f" · device `{active_device}`" if active_device else "")
    )

    st.title("PPE (Personal Protective Equipment) Detection")
    st.caption(
        "Deteksi kepatuhan APD berbasis YOLOv8 — atur sensitivitas dan alarm "
        "di sidebar sesuai kebutuhan area kerja."
    )
    if overrides:
        st.caption(
            "🎚️ Ambang khusus aktif: "
            + ", ".join(f"{CATEGORY_LABELS.get(k, k)} {v:.2f}" for k, v in overrides.items())
        )

    if mode == "Gambar":
        run_image(detector, cfg)
    elif mode == "Video":
        run_video(detector, engine, cfg)
    else:
        run_webcam(detector, engine, cfg)


if __name__ == "__main__":
    main()
