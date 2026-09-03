"""Halaman Streamlit untuk fatigue detection & absensi wajah.

Dipisah dari `app/streamlit_app.py` supaya modul APD tetap satu file yang bisa
dibaca utuh, dan supaya import beratnya (mediapipe + torch + tiga model wajah)
baru terjadi kalau halaman ini benar-benar dibuka.

Tiga tampilan:

    Monitor       kamera/video langsung + kondisi tiap orang
    Karyawan      daftar terdaftar, pendaftaran wajah baru, penghapusan
    Log absensi   riwayat kehadiran + unduh CSV

Yang perlu diingat operator, dan karena itu ditulis di UI-nya juga: level
kelelahan butuh waktu untuk muncul. Beberapa detik pertama setiap sesi akan
menampilkan TIDAK_DIKETAHUI sementara mata orangnya dikalibrasi — itu keadaan
yang benar, bukan kegagalan.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.attendance import AttendanceBook
from src.fatigue.face import FaceDetector, build_embedder
from src.fatigue.fusion import FusionConfig, FusionWeights
from src.fatigue.pipeline import FatiguePipeline, PipelineConfig
from src.fatigue.types import FatigueLevel

SESSION_DIR = PROJECT_ROOT / "outputs" / "sessions"

# Warna badge per level, dipetakan ke token warna Markdown Streamlit supaya
# ikut menyesuaikan tema terang/gelap alih-alih di-hardcode.
LEVEL_COLOR = {
    FatigueLevel.UNKNOWN: "gray",
    FatigueLevel.ALERT: "green",
    FatigueLevel.MILD: "orange",
    FatigueLevel.SEVERE: "red",
    FatigueLevel.CRITICAL: "violet",
}

LEVEL_ICON = {
    FatigueLevel.UNKNOWN: ":material/help:",
    FatigueLevel.ALERT: ":material/check_circle:",
    FatigueLevel.MILD: ":material/visibility:",
    FatigueLevel.SEVERE: ":material/warning:",
    FatigueLevel.CRITICAL: ":material/e911_emergency:",
}

PRESETS = {
    "Seimbang": FusionWeights(),
    "Fokus perilaku": FusionWeights(cnn=0.10, perclos=0.45, blink=0.13,
                                    yawn=0.20, nod=0.12),
    # Preset ini sengaja melonggarkan invarian "hanya PERCLOS yang boleh
    # menaikkan level sendirian" — dengan bobot 0,55 CNN bisa berbicara sendiri.
    # Disediakan karena ada kasus yang memang membutuhkannya (kamera yang
    # sudut matanya buruk sehingga PERCLOS tidak andal), tapi UI memberi tahu
    # konsekuensinya alih-alih membiarkannya ditemukan lewat alarm palsu.
    "Fokus tampilan wajah": FusionWeights(cnn=0.55, perclos=0.25, blink=0.05,
                                          yawn=0.10, nod=0.05),
}

SOURCE_LABELS = {
    "cnn": "tampilan wajah (CNN)",
    "perclos": "PERCLOS",
    "blink": "laju kedip",
    "yawn": "menguap",
    "nod": "kepala terkulai",
}


# --------------------------------------------------------------------------
# Resource — dimuat sekali per proses
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Memuat model wajah…")
def load_pipeline(
    window_seconds: float,
    embedder_backend: str,
    classifier_backend: str,
    use_classifier: bool,
    camera_name: str,
) -> FatiguePipeline:
    """Pipeline di-cache per kombinasi setting.

    `st.cache_resource` (bukan `cache_data`): isinya model dan state, bukan
    data yang bisa di-serialize. Argumennya sengaja semuanya primitif supaya
    Streamlit bisa memakainya sebagai kunci cache — mengubah backend akan
    membangun pipeline baru, bukan diam-diam memakai yang lama.
    """
    config = PipelineConfig(
        window_seconds=window_seconds,
        enable_classifier=use_classifier,
        camera_name=camera_name,
    )
    return FatiguePipeline(
        config=config,
        attendance=load_attendance(embedder_backend),
        embedder_backend=embedder_backend,
        classifier_backend=classifier_backend,
    )


@st.cache_resource(show_spinner=False)
def load_attendance(embedder_backend: str) -> AttendanceBook:
    embedder = build_embedder(embedder_backend)
    return AttendanceBook(backend=embedder.backend, threshold=embedder.threshold)


@st.cache_resource(show_spinner=False)
def load_enrollment_tools(embedder_backend: str):
    """Detektor + embedder khusus pendaftaran (ambang wajah lebih ketat)."""
    return FaceDetector(min_face=60), build_embedder(embedder_backend)


def init_state() -> None:
    st.session_state.setdefault("fatigue_running", False)
    st.session_state.setdefault("fatigue_alerts", [])
    st.session_state.setdefault("fatigue_prev_levels", {})


# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
def render_escalation_notice(weights: FusionWeights, mild: float,
                             severe: float, critical: float) -> None:
    """Beri tahu sumber mana yang bisa menaikkan level sendirian.

    Kombinasi bobot dan ambang menentukan apakah satu sinyal saja sudah cukup
    memicu peringatan. Itu konsekuensi paling penting dari kedua slider di atas,
    dan satu-satunya cara operator biasanya menemukannya adalah lewat alarm
    palsu berhari-hari kemudian. Jadi dikatakan di sini, saat setelannya diubah.
    """
    config = FusionConfig(weights=weights, mild_at=mild, severe_at=severe,
                          critical_at=critical)
    alone = config.sources_that_can_escalate_alone()
    others = [s for s in alone if s != "perclos"]

    if not alone:
        st.caption(
            ":material/shield: Tidak ada sinyal tunggal yang bisa menaikkan "
            "level — semuanya butuh dukungan sinyal lain."
        )
    elif not others:
        st.caption(
            ":material/shield: Hanya PERCLOS yang bisa menaikkan level "
            "sendirian. Sinyal lain butuh dukungan — ini setelan yang disarankan."
        )
    else:
        names = ", ".join(SOURCE_LABELS.get(s, s) for s in others)
        st.warning(
            f"Dengan bobot ini, **{names}** bisa menaikkan level sendirian "
            "tanpa dukungan sinyal apa pun. Model tampilan wajah kadang sangat "
            "yakin dan keliru, jadi harapkan lebih banyak alarm palsu.",
            icon=":material/warning:",
        )


def sidebar_settings() -> dict:
    st.sidebar.title("Fatigue & absensi")
    st.sidebar.caption("YuNet · SFace · MediaPipe FaceMesh · CNN")

    with st.sidebar.expander("Model & backend", expanded=False, icon=":material/memory:"):
        embedder = st.selectbox(
            "Face recognition",
            ["sface", "insightface"],
            help="sface tidak butuh dependency tambahan. insightface lebih akurat "
                 "pada pose miring tapi perlu `pip install insightface onnxruntime`.",
        )
        classifier_backend = st.selectbox(
            "Backend CNN fatigue",
            ["torch", "openvino", "openvino-int8"],
            help="openvino jauh lebih cepat di CPU Intel. Jalankan "
                 "`python scripts/export_fatigue.py` dulu untuk membuat IR-nya.",
        )
        use_classifier = st.toggle(
            "Pakai CNN penampakan wajah", value=True,
            help="Kalau dimatikan, penilaian murni dari sinyal perilaku "
                 "(PERCLOS, kedip, menguap, terkulai).",
        )

    with st.sidebar.expander("Kepekaan", expanded=True, icon=":material/tune:"):
        window = st.slider(
            "Jendela pengamatan (detik)", 15, 180, 60, 5,
            help="Makin panjang makin stabil tapi makin lambat bereaksi. "
                 "60 detik adalah standar untuk PERCLOS.",
        )
        preset_name = st.segmented_control(
            "Prioritas bukti", list(PRESETS), default="Seimbang",
        ) or "Seimbang"
        mild = st.slider("Ambang waspada", 0.05, 0.95, 0.30, 0.05,
                         help="Skor fusi di atas batas ini menaikkan level.")
        severe = st.slider("Ambang lelah", 0.05, 0.95, 0.50, 0.05)
        critical = st.slider("Ambang kritis", 0.05, 0.95, 0.70, 0.05)
        # Ketiganya diurutkan sebelum dipakai: FusionConfig menolak ambang yang
        # tidak menaik, dan tiga slider bebas memungkinkan operator menggeser
        # satu melewati yang lain di tengah jalan. Melempar exception ke layar
        # untuk itu tidak membantu siapa pun.
        mild, severe, critical = sorted((mild, severe, critical))
        if mild == severe or severe == critical:
            # Nilai kembar membuat satu level jadi mustahil dicapai. Diberi
            # jarak minimal satu langkah slider daripada diam-diam dibiarkan.
            severe = min(0.95, max(severe, mild + 0.05))
            critical = min(1.0, max(critical, severe + 0.05))
        dwell = st.slider(
            "Tahan sebelum level turun (detik)", 0, 120, 20, 5,
            help="Level naik seketika, tapi hanya boleh turun setelah kondisinya "
                 "membaik selama ini. Mencegah dashboard berkedip-kedip.",
        )
        render_escalation_notice(PRESETS[preset_name], mild, severe, critical)

    with st.sidebar.expander("Absensi", expanded=False, icon=":material/badge:"):
        camera_name = st.text_input(
            "Nama kamera", value="kamera-1",
            help="Ikut tercatat di log absensi supaya bisa dibedakan per titik.",
        )

    return {
        "embedder": embedder,
        "classifier_backend": classifier_backend,
        "use_classifier": use_classifier,
        "window": float(window),
        "weights": PRESETS[preset_name],
        "mild": mild,
        "severe": severe,
        "critical": critical,
        "dwell": float(dwell),
        "camera_name": camera_name,
    }


def apply_fusion_settings(pipeline: FatiguePipeline, cfg: dict) -> None:
    """Dorong setting sidebar ke pipeline yang di-cache.

    Pipeline-nya `cache_resource`, jadi objeknya bertahan antar-rerun sementara
    slider di sidebar bisa berubah tiap rerun — tanpa langkah ini, mengubah
    ambang tidak berpengaruh apa pun sampai cache-nya dibuang.
    """
    pipeline.configure(
        fusion_config=FusionConfig(
            weights=cfg["weights"],
            mild_at=cfg["mild"],
            severe_at=cfg["severe"],
            critical_at=cfg["critical"],
            downgrade_dwell_seconds=cfg["dwell"],
        ),
        window_seconds=cfg["window"],
        camera_name=cfg["camera_name"],
    )


# --------------------------------------------------------------------------
# Komponen tampilan
# --------------------------------------------------------------------------
def people_frame(people: list) -> pd.DataFrame:
    return pd.DataFrame([{
        "Nama": p.identity.name,
        "Level": p.level.value,
        "Skor": p.score,
        "PERCLOS": p.perclos,
        "Kedip/mnt": round(p.blink_rate, 1),
        "Menguap/mnt": round(p.yawn_rate, 1),
        "Microsleep": p.microsleep_count,
        "Terpejam terlama (dtk)": round(p.longest_closure, 1),
        "Diamati (dtk)": round(p.observed_seconds, 1),
        "Alasan": "; ".join(p.reasons),
    } for p in people])


def render_people(container, people: list) -> None:
    """Kartu ringkas per orang + tabel detail."""
    if not people:
        container.info("Belum ada wajah terdeteksi.", icon=":material/person_search:")
        return

    with container.container(horizontal=True, gap="small"):
        for person in people[:6]:
            with st.container(border=True, width=260):
                color = LEVEL_COLOR[person.level]
                st.markdown(f"**{person.identity.name}**")
                st.markdown(f":{color}-badge[{LEVEL_ICON[person.level]} {person.level.value}]")
                st.progress(
                    min(1.0, person.score),
                    text=f"skor {person.score:.2f} · PERCLOS {person.perclos * 100:.0f}%",
                )
                if person.reasons:
                    st.caption(person.reasons[0])

    frame = people_frame(people)
    container.dataframe(
        frame, hide_index=True, key="fatigue_people",
        column_config={
            "Skor": st.column_config.ProgressColumn(
                "Skor", min_value=0.0, max_value=1.0, format="%.2f"
            ),
            "PERCLOS": st.column_config.ProgressColumn(
                "PERCLOS", min_value=0.0, max_value=1.0, format="percent"
            ),
        },
    )


def collect_alerts(people: list) -> list[dict]:
    """Catat hanya PERUBAHAN level ke WASPADA ke atas.

    Mencatat tiap frame akan menghasilkan ribuan baris untuk satu kejadian dan
    membuat log-nya tidak bisa dibaca.
    """
    new_alerts = []
    previous = st.session_state.fatigue_prev_levels
    for person in people:
        key = person.identity.employee_id or person.identity.name
        if previous.get(key) is person.level:
            continue
        previous[key] = person.level
        if person.level.severity >= FatigueLevel.MILD.severity:
            new_alerts.append({
                "Waktu": datetime.now().strftime("%H:%M:%S"),
                "Nama": person.identity.name,
                "Level": person.level.value,
                "Skor": round(person.score, 3),
                "Alasan": "; ".join(person.reasons),
            })
    return new_alerts


# --------------------------------------------------------------------------
# Monitor
# --------------------------------------------------------------------------
def run_monitor(pipeline: FatiguePipeline, cfg: dict) -> None:
    source = st.segmented_control(
        "Sumber", ["Webcam", "Video", "Gambar"], default="Webcam",
        key="fatigue_source",
    ) or "Webcam"

    if source == "Webcam":
        run_webcam(pipeline, cfg)
    elif source == "Video":
        run_video(pipeline)
    else:
        run_image(pipeline)


def run_webcam(pipeline: FatiguePipeline, cfg: dict) -> None:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        camera_index = st.number_input("Indeks kamera", 0, 8, 0, 1, width=160)
        max_fps = st.slider("Batas FPS", 1, 30, 8, width=220)
        mirror = st.checkbox("Cermin", value=True)
        record = st.checkbox("Rekam sesi")

    with st.container(horizontal=True):
        if st.button("Mulai", type="primary", icon=":material/play_arrow:"):
            st.session_state.fatigue_running = True
            st.session_state.fatigue_alerts = []
            st.session_state.fatigue_prev_levels = {}
        if st.button("Berhenti", icon=":material/stop:"):
            st.session_state.fatigue_running = False

    status_slot = st.empty()
    frame_slot = st.empty()
    people_slot = st.container()
    alert_slot = st.container()

    if not st.session_state.fatigue_running:
        frame_slot.info(
            "Tekan **Mulai** untuk menyalakan kamera. Beberapa detik pertama "
            "dipakai mengkalibrasi mata tiap orang — level akan "
            "`TIDAK_DIKETAHUI` sampai datanya cukup.",
            icon=":material/videocam:",
        )
        render_alert_log(alert_slot)
        return

    cap = cv2.VideoCapture(int(camera_index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        st.session_state.fatigue_running = False
        st.error(f"Gagal membuka kamera index {camera_index}.")
        return

    writer = None
    if record:
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640,
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480)
        path = SESSION_DIR / f"fatigue_{datetime.now():%Y%m%d_%H%M%S}.mp4"
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(max_fps), size
        )
        st.toast(f"Merekam ke {path.name}", icon=":material/fiber_manual_record:")

    min_dt = 1.0 / float(max_fps)
    fps = 0.0
    t_prev = time.perf_counter()
    try:
        while st.session_state.fatigue_running:
            ok, frame = cap.read()
            if not ok:
                st.warning("Gagal membaca frame dari kamera.")
                break
            if mirror:
                frame = cv2.flip(frame, 1)

            analysis = pipeline.process_frame(frame)
            annotated = pipeline.render(frame, analysis)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            st.session_state.fatigue_alerts.extend(collect_alerts(analysis.people))
            del st.session_state.fatigue_alerts[:-200]

            if writer is not None:
                writer.write(annotated)

            frame_slot.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB",
                caption=f"{fps:.1f} fps · {analysis.latency_ms:.0f} ms/frame "
                        f"· {len(analysis.people)} orang",
            )
            worst = analysis.worst_level
            with status_slot.container():
                if worst.severity >= FatigueLevel.SEVERE.severity:
                    st.error(f"Kondisi terburuk: **{worst.value}**",
                             icon=LEVEL_ICON[worst])
                elif worst is FatigueLevel.MILD:
                    st.warning(f"Kondisi terburuk: **{worst.value}**",
                               icon=LEVEL_ICON[worst])
                elif worst is FatigueLevel.ALERT:
                    st.success("Semua terpantau segar", icon=LEVEL_ICON[worst])
                else:
                    st.info("Mengumpulkan data…", icon=LEVEL_ICON[worst])
            render_people(people_slot.empty(), analysis.people)

            elapsed = time.perf_counter() - now
            if elapsed < min_dt:
                time.sleep(min_dt - elapsed)
    finally:
        cap.release()
        if writer is not None:
            writer.release()


def run_video(pipeline: FatiguePipeline) -> None:
    upload = st.file_uploader("Video", type=["mp4", "avi", "mov", "mkv"])
    stride = st.slider(
        "Analisis tiap N frame", 1, 10, 2,
        help="Menaikkan angka ini mempercepat pemrosesan; sinyal kedipan jadi "
             "lebih kasar tapi PERCLOS tetap terukur benar karena dihitung "
             "atas frame yang dianalisis saja.",
    )
    if upload is None:
        return
    if not st.button("Proses video", type="primary", icon=":material/movie:"):
        return

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(upload.name).suffix) as tmp:
        tmp.write(upload.read())
        video_path = tmp.name

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Gagal membuka video.")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    progress = st.progress(0.0, text="Memproses…")
    frame_slot = st.empty()
    people_slot = st.container()

    # Waktu diambil dari posisi frame, bukan jam dinding: pemrosesan yang lebih
    # lambat dari realtime akan merusak semua jendela temporal kalau tidak.
    worst_per_person: dict[str, FatigueLevel] = {}
    index = 0
    last_people: list = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                analysis = pipeline.process_frame(frame, now=index / fps)
                last_people = analysis.people
                for person in analysis.people:
                    key = person.identity.name
                    if person.level.severity > worst_per_person.get(
                        key, FatigueLevel.UNKNOWN
                    ).severity:
                        worst_per_person[key] = person.level
                if index % (stride * 10) == 0:
                    frame_slot.image(
                        cv2.cvtColor(pipeline.render(frame, analysis), cv2.COLOR_BGR2RGB),
                        channels="RGB", caption=f"frame {index}",
                    )
            index += 1
            if total:
                progress.progress(min(1.0, index / total), text=f"{index}/{total} frame")
    finally:
        cap.release()

    progress.empty()
    st.success(f"{index} frame diproses.", icon=":material/check_circle:")
    render_people(people_slot, last_people)
    if worst_per_person:
        st.subheader("Level terburuk per orang")
        st.dataframe(
            pd.DataFrame(
                [{"Nama": k, "Level terburuk": v.value} for k, v in
                 sorted(worst_per_person.items(), key=lambda kv: -kv[1].severity)]
            ),
            hide_index=True,
        )


def run_image(pipeline: FatiguePipeline) -> None:
    st.info(
        "Satu gambar diam tidak bisa menghasilkan level kelelahan — PERCLOS, "
        "kedipan, dan microsleep semuanya butuh waktu. Mode ini untuk mengecek "
        "deteksi wajah dan pengenalan identitas.",
        icon=":material/info:",
    )
    upload = st.file_uploader("Gambar", type=["jpg", "jpeg", "png", "bmp"])
    if upload is None:
        return

    img = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        st.error("File bukan gambar valid.")
        return

    analysis = pipeline.process_frame(img)
    st.image(
        cv2.cvtColor(pipeline.render(img, analysis), cv2.COLOR_BGR2RGB), channels="RGB"
    )
    if not analysis.faces:
        st.warning("Tidak ada wajah terdeteksi.", icon=":material/person_off:")
        return

    st.dataframe(
        pd.DataFrame([{
            "Nama": p.identity.name,
            "Dikenali": p.identity.is_known,
            "Similarity": round(p.identity.similarity, 3),
            "EAR": s.ear,
            "Mata tertutup": s.eye_closed,
            "Mulut terbuka": s.mouth_open,
            "Pitch": None if s.pitch is None else round(s.pitch, 1),
            "Yaw": None if s.yaw is None else round(s.yaw, 1),
            "Skor CNN": round(p.cnn_score, 3),
        } for p, s in zip(analysis.people, analysis.signals)]),
        hide_index=True,
    )


def render_alert_log(container) -> None:
    alerts = st.session_state.fatigue_alerts
    if not alerts:
        return
    container.subheader("Riwayat peringatan sesi ini")
    frame = pd.DataFrame(alerts[::-1])
    container.dataframe(frame, hide_index=True, height=260)
    container.download_button(
        "Unduh CSV", data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"fatigue_alerts_{datetime.now():%Y%m%d_%H%M%S}.csv",
        mime="text/csv", icon=":material/download:",
    )


# --------------------------------------------------------------------------
# Karyawan
# --------------------------------------------------------------------------
def run_employees(book: AttendanceBook, embedder_backend: str) -> None:
    employees = book.list_employees()
    stats = book.stats()

    with st.container(horizontal=True):
        st.metric("Karyawan terdaftar", stats["employees"])
        st.metric("Aktif", stats["active_employees"])
        st.metric("Foto wajah", stats["embeddings"])
        st.metric("Hadir hari ini", stats["present_today"])

    no_photo = [e for e in employees if e.num_embeddings == 0]
    if no_photo:
        st.warning(
            "Belum punya foto wajah dan tidak akan pernah dikenali kamera: "
            + ", ".join(f"`{e.employee_id}`" for e in no_photo),
            icon=":material/no_photography:",
        )

    if employees:
        st.dataframe(
            pd.DataFrame([{
                "ID": e.employee_id, "Nama": e.name, "Departemen": e.department,
                "Foto": e.num_embeddings, "Aktif": e.active,
            } for e in employees]),
            hide_index=True, key="fatigue_employees",
        )
    else:
        st.info("Belum ada karyawan terdaftar.", icon=":material/group_add:")

    st.subheader("Daftarkan wajah")
    st.caption(
        "Unggah 5-10 foto dari sudut dan pencahayaan berbeda. Jumlah dan ragam "
        "foto berpengaruh jauh lebih besar pada keandalan absensi daripada "
        "model yang dipakai."
    )
    # Form dipakai supaya tiap ketikan di kolom nama tidak memicu rerun yang
    # menjalankan ulang deteksi wajah pada semua foto yang sudah diunggah.
    with st.form("fatigue_enroll", border=True):
        with st.container(horizontal=True):
            employee_id = st.text_input("ID karyawan", placeholder="EMP001")
            name = st.text_input("Nama lengkap", placeholder="Budi Santoso")
            department = st.text_input("Departemen", placeholder="Produksi")
        photos = st.file_uploader(
            "Foto wajah", type=["jpg", "jpeg", "png", "bmp"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button(
            "Daftarkan", type="primary", icon=":material/person_add:"
        )

    if submitted:
        enroll(book, embedder_backend, employee_id, name, department, photos)

    with st.expander("Hapus karyawan", icon=":material/delete:"):
        if not employees:
            st.caption("Belum ada karyawan.")
        else:
            target = st.selectbox(
                "Karyawan", [e.employee_id for e in employees],
                format_func=lambda i: f"{i} — "
                                      f"{next(e.name for e in employees if e.employee_id == i)}",
            )
            st.caption(
                "Menghapus akan membuang data biometrik dan seluruh log "
                "kehadirannya secara permanen."
            )
            if st.button("Hapus permanen", type="secondary", icon=":material/delete_forever:"):
                book.delete_employee(target)
                st.toast(f"{target} dihapus.", icon=":material/delete:")
                st.rerun()


def enroll(book, embedder_backend, employee_id, name, department, photos) -> None:
    if not employee_id or not name:
        st.error("ID dan nama wajib diisi.", icon=":material/error:")
        return
    if not photos:
        st.error("Unggah minimal satu foto.", icon=":material/error:")
        return

    detector, embedder = load_enrollment_tools(embedder_backend)
    book.add_employee(employee_id, name, department)

    accepted, rejected = 0, []
    for photo in photos:
        img = cv2.imdecode(np.frombuffer(photo.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            rejected.append((photo.name, "file tidak terbaca"))
            continue
        faces = detector.detect(img)
        if not faces:
            rejected.append((photo.name, "wajah tidak terdeteksi atau terlalu kecil"))
            continue
        if len(faces) > 1:
            rejected.append((photo.name, f"ada {len(faces)} wajah, harus satu orang"))
            continue
        book.add_embedding(employee_id, embedder.embed(img, faces[0]), source=photo.name)
        accepted += 1

    if accepted:
        st.success(f"{accepted} foto terdaftar untuk {name}.",
                   icon=":material/check_circle:")
    else:
        st.error("Tidak ada foto yang bisa dipakai.", icon=":material/error:")
    for filename, reason in rejected:
        st.caption(f"Ditolak — `{filename}`: {reason}")
    if accepted:
        st.rerun()


# --------------------------------------------------------------------------
# Log absensi
# --------------------------------------------------------------------------
def run_attendance(book: AttendanceBook) -> None:
    scope = st.segmented_control(
        "Rentang", ["Hari ini", "7 hari", "Semua"], default="Hari ini",
    ) or "Hari ini"

    if scope == "Hari ini":
        records = book.today()
    elif scope == "7 hari":
        records = book.records(since=time.time() - 7 * 86400, limit=2000)
    else:
        records = book.records(limit=2000)

    if not records:
        st.info("Belum ada catatan kehadiran pada rentang ini.",
                icon=":material/event_busy:")
        return

    frame = pd.DataFrame([r.to_row() for r in records])
    with st.container(horizontal=True):
        st.metric("Catatan", len(frame))
        st.metric("Orang unik", frame["employee_id"].nunique())

    st.dataframe(frame, hide_index=True, key="fatigue_attendance")
    st.download_button(
        "Unduh CSV", data=frame.to_csv(index=False).encode("utf-8"),
        file_name=f"absensi_{datetime.now():%Y%m%d}.csv",
        mime="text/csv", icon=":material/download:",
    )

    if scope != "Hari ini":
        by_day = (
            pd.to_datetime(frame["waktu"]).dt.date.value_counts().sort_index()
            .rename_axis("tanggal").reset_index(name="kehadiran")
        )
        st.bar_chart(by_day, x="tanggal", y="kehadiran")


# --------------------------------------------------------------------------
def render() -> None:
    """Titik masuk halaman, dipanggil dari app/streamlit_app.py."""
    init_state()
    cfg = sidebar_settings()

    st.title("Deteksi fatigue pekerja & absensi wajah")
    st.caption(
        "Wajah dikenali untuk absensi, lalu kondisinya dinilai dari gabungan "
        "penampakan (CNN) dan perilaku mata/mulut/kepala sepanjang waktu."
    )

    try:
        pipeline = load_pipeline(
            cfg["window"], cfg["embedder"], cfg["classifier_backend"],
            cfg["use_classifier"], cfg["camera_name"],
        )
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as exc:
        st.error(str(exc), icon=":material/error:")
        st.stop()

    apply_fusion_settings(pipeline, cfg)
    book = load_attendance(cfg["embedder"])

    if pipeline.classifier is None:
        st.warning(
            "CNN fatigue tidak aktif — penilaian memakai sinyal perilaku saja. "
            "Latih dulu dengan `python scripts/prepare_fatigue_dataset.py` lalu "
            "`python scripts/train_fatigue.py`.",
            icon=":material/model_training:",
        )

    view = st.segmented_control(
        "Tampilan", ["Monitor", "Karyawan", "Log absensi"], default="Monitor",
        key="fatigue_view", label_visibility="collapsed",
    ) or "Monitor"

    if view == "Monitor":
        run_monitor(pipeline, cfg)
    elif view == "Karyawan":
        run_employees(book, cfg["embedder"])
    else:
        run_attendance(book)
