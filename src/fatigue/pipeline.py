"""Orkestrasi satu frame: wajah -> identitas -> sinyal -> level kelelahan.

Ini kelas yang dipakai CLI, FastAPI, dan Streamlit. Semua komponen berat
(detektor, embedder, landmarker, classifier) dimiliki di sini dan dibuat sekali
saja; pemanggil cukup memberi frame.

Satu keputusan desain yang menentukan perilakunya di lapangan: **state fatigue
diikat ke identitas, bukan ke posisi kotak wajah.** Tracker berbasis IoU biasa
akan kehilangan riwayat seseorang setiap kali ia keluar-masuk frame atau
tertukar dengan rekan di sebelahnya, dan riwayat 60 detik yang hilang berarti
sistemnya buta lagi selama satu menit. Karena wajahnya toh sudah di-embed untuk
absensi, mengikat state ke identitas hasil pengenalan itu gratis dan jauh lebih
stabil. Orang yang belum terdaftar tetap dilacak, memakai identitas sementara
berbasis kemiripan embedding antar-frame.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np

from src.fatigue.attendance import AttendanceBook, AttendanceRecord
from src.fatigue.classifier import FatigueClassifier, build_classifier
from src.fatigue.face import FaceDetector, FaceEmbedder, build_embedder
from src.fatigue.fusion import FatigueFusion, FusionConfig
from src.fatigue.landmarks import FaceLandmarker
from src.fatigue.records import FatigueLog
from src.fatigue.temporal import PersonTracker
from src.fatigue.types import (
    LEVEL_COLORS,
    FaceBox,
    FatigueLevel,
    FrameAnalysis,
    Identity,
    PersonState,
)

# Kemiripan minimum untuk menganggap wajah tak-dikenal di frame ini adalah
# orang yang sama dengan track tak-dikenal yang sudah ada. Lebih ketat dari
# ambang absensi: salah menggabungkan dua orang asing berarti mencampur
# riwayat mata mereka, dan itu merusak PERCLOS keduanya.
UNKNOWN_TRACK_THRESHOLD = 0.55
# Track tak-dikenal yang tidak terlihat selama ini dibuang.
UNKNOWN_TRACK_TTL = 90.0


@dataclass
class PipelineConfig:
    """Semua tombol yang bisa diputar tanpa menyentuh kode."""

    face_conf: float = 0.6
    min_face: int = 40
    max_faces: int = 8
    crop_margin: float = 0.25
    # Classifier CNN tidak perlu jalan tiap frame: penampakan wajah berubah
    # jauh lebih lambat daripada 25 fps, dan ia komponen termahal di pipeline.
    # 1 dari 5 frame sudah memberi ~5 pembaruan/detik — jauh lebih rapat
    # daripada yang dibutuhkan jendela 60 detik.
    classifier_every: int = 5
    # Begitu juga embedding wajah: identitas seseorang tidak berubah di antara
    # dua frame. Di sela-selanya, wajah dicocokkan ke track lewat tumpang-tindih
    # kotak (IoU) yang ongkosnya nol. Embedding tetap dihitung untuk wajah yang
    # BARU muncul, berapa pun nomor frame-nya — kalau tidak, orang yang baru
    # masuk ruangan akan tercatat "tidak dikenal" sampai penyegaran berikutnya.
    embedder_every: int = 10
    # IoU minimum untuk menganggap kotak wajah frame ini lanjutan track lama.
    iou_match_threshold: float = 0.35
    # Umur maksimum kotak track yang masih boleh dipakai untuk pencocokan IoU.
    # Lebih tua dari ini, posisinya sudah tidak bisa dipercaya.
    iou_max_age: float = 2.0
    enable_attendance: bool = True
    # Simpan riwayat fatigue ke SQLite supaya bisa dilaporkan belakangan.
    # Tanpa ini, PERCLOS dan microsleep hilang begitu aplikasi ditutup — dan
    # pertanyaan paling wajar seorang supervisor ("minggu lalu siapa yang
    # paling sering mengantuk?") tidak bisa dijawab sama sekali.
    enable_recording: bool = True
    # CNN MATI secara default. Diukur pada tiga domain: di data latihnya sendiri
    # (foto internet) ia benar — rata-rata p(lelah) 0,12 untuk kelas non-lelah.
    # Pada wajah pekerja di foto lapangan nyata, rata-ratanya melompat ke 0,57
    # dan 59% orang biasa ditandai lelah. Pada wajah webcam yang diuji, ia
    # memberi 0,90-0,99 secara konstan di lima foto berbeda rentang enam bulan
    # — itu konstanta, bukan pengukuran.
    #
    # Sinyal perilaku (PERCLOS, microsleep, kedip, menguap, terkulai) tidak
    # punya masalah ini: semuanya pengukuran fisik yang tidak peduli bentuk
    # wajah siapa pun. Jadi default-nya berpihak ke sana.
    #
    # Nyalakan lagi (`--classifier`, atau FATIGUE_CLASSIFIER=1) setelah model
    # dilatih ulang dengan frame dari kamera Anda sendiri — di domain itu ia
    # bisa jadi berguna, dan `scripts/train_fatigue.py` siap menerimanya.
    enable_classifier: bool = False
    window_seconds: float = 60.0
    camera_name: str = ""


@dataclass
class _Track:
    """State satu orang yang sedang diamati."""

    key: str
    identity: Identity
    tracker: PersonTracker
    fusion: FatigueFusion
    embedding: np.ndarray
    last_cnn_score: float = 0.0
    last_seen: float = 0.0
    bbox: list[int] | None = None
    state: PersonState = field(default_factory=lambda: PersonState(Identity.unknown()))


def iou(a: list[int], b: list[int]) -> float:
    """Intersection-over-union dua kotak [x1, y1, x2, y2]."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class FatiguePipeline:
    """Analisis fatigue + absensi untuk aliran frame dari satu kamera."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        fusion_config: FusionConfig | None = None,
        attendance: AttendanceBook | None = None,
        embedder_backend: str | None = None,
        classifier_backend: str | None = None,
        classifier: FatigueClassifier | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.fusion_config = fusion_config or FusionConfig()

        self.detector = FaceDetector(
            conf=self.config.face_conf, min_face=self.config.min_face
        )
        self.landmarker = FaceLandmarker(margin=self.config.crop_margin)

        self.embedder: FaceEmbedder | None = None
        self.attendance: AttendanceBook | None = None
        if self.config.enable_attendance:
            self.embedder = build_embedder(embedder_backend)
            self.attendance = attendance or AttendanceBook(
                backend=self.embedder.backend, threshold=self.embedder.threshold
            )

        # `enable_classifier` berarti "muatkan satu untukku dari disk".
        # Menyerahkan objek `classifier` secara eksplisit selalu dihormati —
        # pemanggil yang repot membuatnya jelas ingin ia dipakai, dan
        # mengabaikannya diam-diam karena sebuah flag adalah jebakan.
        self.classifier: FatigueClassifier | None = classifier
        if self.classifier is None and self.config.enable_classifier:
            try:
                self.classifier = build_classifier(classifier_backend)
            except FileNotFoundError as exc:
                # Pipeline tetap berguna tanpa CNN — sinyal perilaku saja
                # sudah menangkap mayoritas kasus, dan justru lebih dapat
                # dipercaya. Menolak jalan sama sekali hanya karena checkpoint
                # belum dilatih akan mematikan fitur absensi tanpa alasan.
                print(f"[WARN] Classifier fatigue tidak aktif: {exc}")
                self.classifier = None

        # Kalau tidak ada classifier, bobot CNN dibuang dan sisanya
        # dinormalisasi ulang. Membiarkannya menyumbang nol akan menyusutkan
        # skala skor diam-diam — skor maksimum jadi 0,80 sehingga ambang KRITIS
        # 0,70 nyaris mustahil tercapai, dan sistemnya jadi tumpul tanpa ada
        # yang menyadarinya.
        if self.classifier is None:
            self.fusion_config = replace(
                self.fusion_config, weights=self.fusion_config.weights.without("cnn")
            )

        self.log: FatigueLog | None = None
        if self.config.enable_recording:
            db = self.attendance.db_path if self.attendance is not None else None
            self.log = FatigueLog(db_path=db, camera=self.config.camera_name)

        self._tracks: dict[str, _Track] = {}
        self._frame_index = 0
        self._unknown_counter = 0
        self.recent_checkins: list[AttendanceRecord] = []
        self.recent_events: list = []

    # ---------- utilitas ----------
    @property
    def num_tracked(self) -> int:
        """Berapa orang yang riwayatnya sedang dipegang pipeline ini."""
        return len(self._tracks)

    def configure(
        self,
        fusion_config: FusionConfig | None = None,
        window_seconds: float | None = None,
        camera_name: str | None = None,
    ) -> None:
        """Ubah kebijakan penilaian tanpa membuang riwayat yang sudah terkumpul.

        Setting baru harus didorong ke setiap orang yang sedang dilacak, bukan
        cuma disimpan di config — objek `FatigueFusion` dan `PersonTracker`
        milik tiap track memegang salinannya sendiri dan tidak membacanya ulang.

        Riwayat temporalnya sengaja dipertahankan: menaikkan ambang bukan
        alasan untuk melupakan apa yang sudah diamati satu menit terakhir.
        """
        if fusion_config is not None:
            # Bobot CNN dibuang lagi kalau classifier-nya memang tidak ada —
            # sidebar UI mengirim bobot lengkap tiap rerun, dan tanpa langkah
            # ini setelan pertama akan benar lalu diam-diam rusak begitu
            # operator menggeser slider apa pun.
            if self.classifier is None:
                fusion_config = replace(
                    fusion_config, weights=fusion_config.weights.without("cnn")
                )
            self.fusion_config = fusion_config
            for track in self._tracks.values():
                track.fusion.config = fusion_config
        if window_seconds is not None:
            self.config.window_seconds = window_seconds
            for track in self._tracks.values():
                track.tracker.window_seconds = window_seconds
        if camera_name is not None:
            self.config.camera_name = camera_name

    def describe(self) -> dict:
        return {
            "face_detector": self.detector.model_path,
            "landmarker": self.landmarker.model_path,
            "embedder": self.embedder.backend if self.embedder else None,
            "classifier": self.classifier.describe() if self.classifier else None,
            "attendance_db": str(self.attendance.db_path) if self.attendance else None,
            "window_seconds": self.config.window_seconds,
        }

    def crop_face(self, frame: np.ndarray, face: FaceBox) -> np.ndarray:
        """Crop wajah dengan margin yang SAMA dengan yang dipakai saat training.

        Satu definisi dipakai landmarker maupun classifier; kalau keduanya
        pakai margin berbeda, salah satunya pasti tidak cocok dengan datanya.
        """
        return self.landmarker.crop(frame, face)

    # ---------- pencocokan track ----------
    def _match_by_iou(self, faces: list[FaceBox], now: float) -> list[str | None]:
        """Kunci track untuk tiap wajah berdasar tumpang-tindih kotak.

        Greedy dari pasangan ber-IoU tertinggi, dan satu track hanya boleh
        dipakai satu wajah. Tanpa aturan eksklusif itu, dua orang yang saling
        melintas bisa sama-sama mengklaim track yang sama dan riwayat mata
        mereka tercampur.
        """
        result: list[str | None] = [None] * len(faces)
        if not self._tracks:
            return result

        candidates = [
            (iou(face.bbox, track.bbox), i, key)
            for i, face in enumerate(faces)
            for key, track in self._tracks.items()
            if track.bbox is not None and now - track.last_seen <= self.config.iou_max_age
        ]
        candidates = [c for c in candidates if c[0] >= self.config.iou_match_threshold]
        candidates.sort(reverse=True)

        taken_faces: set[int] = set()
        taken_tracks: set[str] = set()
        for _, i, key in candidates:
            if i in taken_faces or key in taken_tracks:
                continue
            result[i] = key
            taken_faces.add(i)
            taken_tracks.add(key)
        return result

    def _resolve_track(
        self,
        identity: Identity | None,
        embedding: np.ndarray | None,
        now: float,
        hint: str | None = None,
    ) -> _Track:
        """Cari (atau buat) track untuk orang ini.

        `identity` None berarti frame ini melewatkan pengenalan (dicocokkan
        lewat IoU saja) — identitas track yang sudah ada dipertahukan apa
        adanya, bukan diturunkan jadi "tidak dikenal".
        """
        if identity is not None and identity.is_known and identity.employee_id:
            key = f"emp:{identity.employee_id}"
        elif hint is not None:
            key = hint
        elif embedding is not None:
            key = self._match_unknown(embedding, now)
        else:
            # Tanpa embedder, semua wajah tak-dikenal jatuh ke satu track.
            # Cukup untuk mode satu-orang (webcam operator), dan pembatasan
            # ini eksplisit di dokumentasi mode tersebut.
            key = "anon:0"

        # Orang yang tadinya anon lalu ternyata karyawan terdaftar: riwayat
        # matanya dipindahkan ke track karyawan, tidak dibuang. Menit-menit
        # awal itu justru sering yang paling informatif.
        #
        # Migrasi HANYA dari track anonim. Memindahkan riwayat dari satu track
        # karyawan ke karyawan lain berarti mencampur PERCLOS dan kalibrasi
        # mata dua orang berbeda — dan itu justru terjadi persis ketika sistem
        # salah mengenali, saat datanya paling tidak boleh dipercaya.
        if (
            hint is not None
            and hint != key
            and hint.startswith("anon:")
            and hint in self._tracks
            and key not in self._tracks
        ):
            migrating = self._tracks.pop(hint)
            migrating.key = key
            self._tracks[key] = migrating

        track = self._tracks.get(key)
        if track is None:
            track = _Track(
                key=key,
                identity=identity or Identity.unknown(),
                tracker=PersonTracker(window_seconds=self.config.window_seconds),
                fusion=FatigueFusion(self.fusion_config),
                embedding=(embedding if embedding is not None
                           else np.zeros(0, dtype=np.float32)),
            )
            self._tracks[key] = track
        else:
            # Orang yang tadinya tak dikenal lalu berhasil dikenali: identitas
            # diperbarui, riwayat matanya dipertahankan. Sebaliknya, satu frame
            # yang gagal mengenali TIDAK menghapus identitas yang sudah mantap —
            # wajah yang sekilas menoleh tidak boleh membuat namanya berkedip
            # hilang-muncul di dashboard.
            if identity is not None and identity.is_known:
                track.identity = identity
            if embedding is not None:
                if track.embedding.shape != embedding.shape:
                    track.embedding = embedding
                else:
                    # Embedding referensi diperbarui perlahan supaya track
                    # mengikuti perubahan pencahayaan tanpa hanyut ke orang lain.
                    track.embedding = 0.9 * track.embedding + 0.1 * embedding
                    norm = float(np.linalg.norm(track.embedding))
                    if norm > 0:
                        track.embedding /= norm
        track.last_seen = now
        return track

    def _match_unknown(self, embedding: np.ndarray, now: float) -> str:
        """Kunci track untuk wajah tak dikenal, dicocokkan ke track anon lama."""
        best_key, best_sim = None, UNKNOWN_TRACK_THRESHOLD
        for key, track in self._tracks.items():
            if not key.startswith("anon:") or track.embedding.shape != embedding.shape:
                continue
            sim = float(np.dot(track.embedding, embedding))
            if sim > best_sim:
                best_key, best_sim = key, sim
        if best_key is not None:
            return best_key
        self._unknown_counter += 1
        return f"anon:{self._unknown_counter}"

    def _evict_tracks(self, now: float) -> None:
        """Buang track anon yang sudah lama hilang; track karyawan dipertahankan.

        Track karyawan disimpan (riwayatnya di-reset saja) karena kalibrasi
        ambang mata personalnya mahal untuk dibangun ulang dan tidak berubah
        selama shift.
        """
        for key in [k for k in self._tracks if k.startswith("anon:")]:
            if now - self._tracks[key].last_seen > UNKNOWN_TRACK_TTL:
                del self._tracks[key]
        for track in self._tracks.values():
            if track.key.startswith("emp:") and now - track.last_seen > UNKNOWN_TRACK_TTL:
                track.tracker.reset()
                track.fusion.reset()

    # ---------- inti ----------
    def process_frame(self, frame: np.ndarray, now: float | None = None) -> FrameAnalysis:
        """Analisis satu frame. Aman dipanggil berulang dengan frame berurutan."""
        t_start = time.perf_counter()
        t = time.time() if now is None else now
        self._frame_index += 1

        h, w = frame.shape[:2]
        analysis = FrameAnalysis(width=w, height=h)

        faces = self.detector.detect(frame)[: self.config.max_faces]
        analysis.faces = faces
        if not faces:
            self._evict_tracks(t)
            analysis.latency_ms = (time.perf_counter() - t_start) * 1000
            return analysis

        # --- asosiasi track: IoU dulu (gratis), embedding hanya kalau perlu ---
        assigned = self._match_by_iou(faces, t)
        refresh = self._frame_index % max(1, self.config.embedder_every) == 0

        embeddings: list[np.ndarray | None] = [None] * len(faces)
        identities: list[Identity | None] = [None] * len(faces)
        if self.embedder is not None:
            for i, face in enumerate(faces):
                if assigned[i] is not None and not refresh:
                    continue
                try:
                    embeddings[i] = self.embedder.embed(frame, face)
                except ValueError:
                    embeddings[i] = None
                if embeddings[i] is not None and self.attendance is not None:
                    identities[i] = self.attendance.identify(embeddings[i])

        # --- sinyal perilaku ---
        signals = [self.landmarker.analyze(frame, face) for face in faces]
        analysis.signals = signals

        # --- skor CNN (di-subsample) ---
        run_cnn = (
            self.classifier is not None
            and self._frame_index % max(1, self.config.classifier_every) == 0
        )
        cnn_scores: list[float | None] = [None] * len(faces)
        if run_cnn:
            crops = [self.crop_face(frame, f) for f in faces]
            valid = [i for i, c in enumerate(crops) if c.size]
            if valid:
                probs = self.classifier.predict_batch([crops[i] for i in valid])
                for i, p in zip(valid, probs.tolist()):
                    cnn_scores[i] = p

        # --- agregasi & fusi per orang ---
        for i, face in enumerate(faces):
            track = self._resolve_track(
                identities[i], embeddings[i], t, hint=assigned[i]
            )
            track.bbox = face.bbox
            if cnn_scores[i] is not None:
                track.last_cnn_score = cnn_scores[i]

            track.tracker.update(signals[i], track.last_cnn_score, now=t)
            summary = track.tracker.summarize(now=t)
            result = track.fusion.update(summary, now=t)

            track.state = PersonState(
                identity=track.identity,
                level=result.level,
                score=result.score,
                cnn_score=track.last_cnn_score,
                perclos=summary.perclos,
                blink_rate=summary.blink_rate,
                yawn_rate=summary.yawn_rate,
                nod_rate=summary.nod_rate,
                microsleep_count=summary.microsleep_count,
                longest_closure=summary.longest_closure,
                observed_seconds=summary.observed_seconds,
                reasons=result.reasons,
                updated_at=t,
            )
            analysis.people.append(track.state)

            # --- absensi ---
            # Dicatat dari `identities[i]`, bukan `track.identity`: hanya frame
            # yang benar-benar menjalankan pengenalan yang boleh membuat baris
            # absensi. Frame yang cuma dicocokkan lewat IoU mewarisi nama, dan
            # mewarisi nama bukan bukti kehadiran.
            if self.attendance is not None and identities[i] is not None and identities[i].is_known:
                record = self.attendance.check_in(
                    identities[i], camera=self.config.camera_name, now=t
                )
                if record is not None:
                    self.recent_checkins.append(record)
                    # Hanya 50 terakhir yang ditahan di memori; riwayat penuh
                    # ada di database dan tidak perlu diduplikasi di sini.
                    del self.recent_checkins[:-50]

        # Pencatatan dilakukan sekali per frame untuk semua orang sekaligus:
        # `FatigueLog` sendiri yang memutuskan mana yang layak ditulis, jadi
        # ketiga pemanggil (CLI, UI, API) mendapat kebijakan yang sama.
        if self.log is not None and analysis.people:
            new_events = self.log.record(analysis.people, now=t)
            if new_events:
                self.recent_events.extend(new_events)
                del self.recent_events[:-50]

        self._evict_tracks(t)
        analysis.latency_ms = (time.perf_counter() - t_start) * 1000
        return analysis

    # ---------- rendering ----------
    @staticmethod
    def render(frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        """Gambar kotak wajah + nama + level di atas salinan frame."""
        out = frame.copy()
        for face, person in zip(analysis.faces, analysis.people):
            x1, y1, x2, y2 = face.bbox
            color = LEVEL_COLORS[person.level]
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            name = person.identity.name if person.identity.is_known else "?"
            text = f"{name} | {person.level.value}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            if person.level.severity >= FatigueLevel.MILD.severity:
                sub = f"PERCLOS {person.perclos * 100:.0f}%  skor {person.score:.2f}"
                cv2.putText(out, sub, (x1 + 2, y2 + 14), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, color, 1, cv2.LINE_AA)
        return out

    @staticmethod
    def draw_hud(
        frame: np.ndarray, analysis: FrameAnalysis, fps: float | None = None
    ) -> np.ndarray:
        """Panel ringkas di pojok kiri atas, sejalan dengan HUD PPE."""
        out = frame
        pad, line_h = 8, 18
        rows = analysis.people[:6]
        height = line_h * (len(rows) + 1) + pad * 2

        overlay = out.copy()
        cv2.rectangle(overlay, (0, 0), (330, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.55, out, 0.45, 0, out)

        header = f"FATIGUE  {len(analysis.people)} orang"
        if fps is not None:
            header += f"  {fps:4.1f} fps"
        cv2.putText(out, header, (pad, pad + 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)

        for i, person in enumerate(rows, start=1):
            name = person.identity.name if person.identity.is_known else "tidak dikenal"
            line = f"{name}: {person.level.value} ({person.score:.2f})"
            cv2.putText(out, line, (pad, pad + 12 + i * line_h),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        LEVEL_COLORS[person.level], 1, cv2.LINE_AA)
        return out

    def close(self) -> None:
        self.landmarker.close()


def build_pipeline(**kwargs) -> FatiguePipeline:
    """Bangun pipeline dengan default dari environment."""
    config = PipelineConfig(
        enable_attendance=os.getenv("FATIGUE_ATTENDANCE", "1") != "0",
        enable_classifier=os.getenv("FATIGUE_CLASSIFIER", "0") == "1",
        enable_recording=os.getenv("FATIGUE_RECORDING", "1") != "0",
        camera_name=os.getenv("CAMERA_NAME", ""),
    )
    return FatiguePipeline(config=config, **kwargs)
