"""Test pipeline fatigue: asosiasi track, subsampling, dan integrasi absensi.

Yang diuji di sini adalah *perekatnya* — bagian yang menentukan riwayat siapa
masuk ke mana. Salah di situ tidak memunculkan exception apa pun; ia hanya
membuat PERCLOS dua orang tercampur, dan itu jenis bug yang paling sulit
terlihat dari layar.

Komponen berat (YuNet, SFace, MediaPipe, CNN) di-stub. Kualitas modelnya bukan
urusan test ini dan sudah diukur `scripts/evaluate_fatigue.py`; yang penting
di sini adalah logika di antaranya.
"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.fatigue import pipeline as pipeline_module
from src.fatigue.attendance import AttendanceBook
from src.fatigue.pipeline import FatiguePipeline, PipelineConfig, iou
from src.fatigue.types import FaceBox, FatigueSignals, Identity

DIM = 128


def unit(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=DIM).astype(np.float32)
    return v / np.linalg.norm(v)


class FakeDetector:
    """Mengembalikan kotak wajah yang di-skrip per frame."""

    def __init__(self) -> None:
        self.script: list[list[list[int]]] = []
        self.index = 0

    def detect(self, frame):
        boxes = self.script[min(self.index, len(self.script) - 1)] if self.script else []
        self.index += 1
        return [
            FaceBox(bbox=list(b), confidence=0.9, landmarks=[[0, 0]] * 5,
                    raw=[0.0] * 15)
            for b in boxes
        ]


class FakeEmbedder:
    """Memberi vektor identitas berdasarkan urutan wajah di frame."""

    backend = "sface"
    dim = DIM

    def __init__(self, vectors: list[np.ndarray]) -> None:
        self.vectors = vectors
        self.calls = 0
        self._order = 0

    @property
    def threshold(self) -> float:
        return 0.40

    def embed(self, frame, face):
        self.calls += 1
        # Wajah di-urut berdasarkan koordinat x agar pemetaannya stabil.
        index = min(face.bbox[0] // 100, len(self.vectors) - 1)
        return self.vectors[index]


class FakeLandmarker:
    """Selalu melaporkan mata terbuka, wajah menghadap depan."""

    model_path = "fake"
    margin = 0.25

    def __init__(self) -> None:
        self.calls = 0

    def crop(self, frame, face):
        x1, y1, x2, y2 = face.bbox
        return frame[max(0, y1):y2, max(0, x1):x2]

    def analyze(self, frame, face):
        self.calls += 1
        return FatigueSignals(ear=0.30, mar=0.05, eye_closed=False,
                              mouth_open=False, pitch=0.0, yaw=0.0, roll=0.0,
                              blink_score=0.05, jaw_open=0.02)

    def close(self):
        pass


class FakeClassifier:
    backend = "fake"
    threshold = 0.5

    def __init__(self, score: float = 0.2) -> None:
        self.score = score
        self.calls = 0

    def predict_batch(self, crops):
        self.calls += 1
        return np.full(len(crops), self.score, dtype=np.float32)

    def describe(self):
        return "fake classifier"


@pytest.fixture()
def rig(tmp_path, monkeypatch):
    """Pipeline dengan seluruh komponen berat di-stub."""
    detector = FakeDetector()
    embedder = FakeEmbedder([unit(1), unit(2), unit(3)])
    landmarker = FakeLandmarker()

    monkeypatch.setattr(pipeline_module, "FaceDetector", lambda **kw: detector)
    monkeypatch.setattr(pipeline_module, "FaceLandmarker", lambda **kw: landmarker)
    monkeypatch.setattr(pipeline_module, "build_embedder", lambda b=None: embedder)

    book = AttendanceBook(db_path=tmp_path / "att.db", threshold=0.40,
                          reentry_gap=300.0)
    classifier = FakeClassifier()
    pipe = FatiguePipeline(
        config=PipelineConfig(window_seconds=30.0, classifier_every=1,
                              embedder_every=10),
        attendance=book,
        classifier=classifier,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return pipe, detector, embedder, landmarker, classifier, book, frame


# ---------- IoU ----------
def test_iou_identical_boxes():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)


def test_iou_disjoint_boxes():
    assert iou([0, 0, 10, 10], [50, 50, 60, 60]) == 0.0


def test_iou_partial_overlap():
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3, abs=1e-6)


# ---------- asosiasi ----------
def test_same_person_keeps_one_track(rig):
    pipe, detector, _, _, _, _, frame = rig
    # Kotak bergeser sedikit tiap frame, seperti orang yang bergerak pelan.
    detector.script = [[[100 + i, 100, 200 + i, 200]] for i in range(20)]
    for i in range(20):
        pipe.process_frame(frame, now=i * 0.1)
    assert pipe.num_tracked == 1


def test_two_people_get_two_tracks(rig):
    pipe, detector, _, _, _, _, frame = rig
    detector.script = [[[0, 100, 90, 200], [200, 100, 290, 200]]] * 10
    for i in range(10):
        analysis = pipe.process_frame(frame, now=i * 0.1)
    assert pipe.num_tracked == 2
    assert len(analysis.people) == 2


def test_history_survives_a_brief_disappearance(rig):
    """Orang yang tertutup rekannya sesaat tidak boleh kehilangan riwayat."""
    pipe, detector, _, _, _, _, frame = rig
    box = [[100, 100, 200, 200]]
    detector.script = [box] * 30 + [[]] * 3 + [box] * 30

    for i in range(63):
        analysis = pipe.process_frame(frame, now=i * 0.1)
    assert pipe.num_tracked == 1
    # Jendela pengamatan tetap panjang, bukan mulai lagi dari nol.
    assert analysis.people[0].observed_seconds > 5.0


# ---------- subsampling ----------
def test_embedding_is_subsampled_not_run_every_frame(rig):
    pipe, detector, embedder, _, _, _, frame = rig
    detector.script = [[[100, 100, 200, 200]]] * 30
    for i in range(30):
        pipe.process_frame(frame, now=i * 0.1)
    # embedder_every=10 -> frame 1 (wajah baru) + frame 10, 20, 30.
    assert embedder.calls <= 5, f"embedding jalan {embedder.calls} kali dari 30 frame"


def test_new_face_is_embedded_immediately(rig):
    """Orang yang baru masuk tidak boleh menunggu siklus penyegaran.

    Kalau ia menunggu, ia tercatat 'tidak dikenal' selama beberapa detik dan
    absensinya bisa terlewat sama sekali.
    """
    pipe, detector, embedder, _, _, _, frame = rig
    one = [[100, 100, 200, 200]]
    two = [[100, 100, 200, 200], [300, 100, 400, 200]]
    detector.script = [one] * 3 + [two] * 3

    pipe.process_frame(frame, now=0.0)
    calls_before = embedder.calls
    for i in range(1, 3):
        pipe.process_frame(frame, now=i * 0.1)
    assert embedder.calls == calls_before      # tidak ada wajah baru

    pipe.process_frame(frame, now=0.3)          # orang kedua muncul
    assert embedder.calls > calls_before


def test_landmarks_run_every_frame(rig):
    """PERCLOS butuh resolusi waktu penuh — landmark TIDAK boleh di-subsample."""
    pipe, detector, _, landmarker, _, _, frame = rig
    detector.script = [[[100, 100, 200, 200]]] * 20
    for i in range(20):
        pipe.process_frame(frame, now=i * 0.1)
    assert landmarker.calls == 20


def test_classifier_respects_its_interval(rig, tmp_path, monkeypatch):
    detector = FakeDetector()
    detector.script = [[[100, 100, 200, 200]]] * 20
    monkeypatch.setattr(pipeline_module, "FaceDetector", lambda **kw: detector)
    monkeypatch.setattr(pipeline_module, "FaceLandmarker", lambda **kw: FakeLandmarker())
    monkeypatch.setattr(pipeline_module, "build_embedder",
                        lambda b=None: FakeEmbedder([unit(1)]))

    classifier = FakeClassifier()
    pipe = FatiguePipeline(
        config=PipelineConfig(classifier_every=5, enable_attendance=False),
        classifier=classifier,
    )
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    for i in range(20):
        pipe.process_frame(frame, now=i * 0.1)
    assert classifier.calls == 4        # frame 5, 10, 15, 20


# ---------- absensi ----------
def test_recognized_person_is_checked_in(rig):
    pipe, detector, embedder, _, _, book, frame = rig
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", embedder.vectors[1])   # wajah di kolom x=100..

    detector.script = [[[100, 100, 200, 200]]] * 5
    for i in range(5):
        analysis = pipe.process_frame(frame, now=1000.0 + i * 0.1)

    assert analysis.people[0].identity.employee_id == "EMP001"
    assert len(pipe.recent_checkins) == 1
    assert len(book.records()) == 1


def test_iou_only_frames_do_not_create_extra_attendance_rows(rig):
    """Mewarisi nama lewat IoU bukan bukti kehadiran baru."""
    pipe, detector, embedder, _, _, book, frame = rig
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", embedder.vectors[1])

    detector.script = [[[100, 100, 200, 200]]] * 60
    for i in range(60):
        pipe.process_frame(frame, now=1000.0 + i * 0.1)
    assert len(book.records()) == 1


def test_identity_is_not_lost_between_refresh_frames(rig):
    """Frame yang tidak menjalankan pengenalan tetap menampilkan nama."""
    pipe, detector, embedder, _, _, book, frame = rig
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", embedder.vectors[1])

    detector.script = [[[100, 100, 200, 200]]] * 9
    names = []
    for i in range(9):
        analysis = pipe.process_frame(frame, now=1000.0 + i * 0.1)
        names.append(analysis.people[0].identity.name)
    assert set(names) == {"Budi"}, names


def test_history_is_not_mixed_between_two_employees(rig):
    """Salah kenal tidak boleh memindahkan riwayat mata satu orang ke orang lain.

    Ini justru terjadi persis ketika sistem sedang salah — saat datanya paling
    tidak boleh dipercaya. Track karyawan harus tetap terpisah.
    """
    pipe, detector, embedder, _, _, book, frame = rig
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", embedder.vectors[1])
    book.add_employee("EMP002", "Ani")
    book.add_embedding("EMP002", embedder.vectors[2])

    # Budi diamati lama di posisi kiri.
    detector.script = [[[100, 100, 200, 200]]] * 20
    for i in range(20):
        pipe.process_frame(frame, now=1000.0 + i * 0.1)
    budi = pipe._tracks["emp:EMP001"]           # noqa: SLF001
    budi_observed = budi.tracker.summarize(now=1001.9).observed_seconds
    assert budi_observed > 1.0

    # Frame berikutnya kotaknya tumpang-tindih (IoU cocok ke track Budi), tapi
    # embedding-nya ternyata Ani.
    detector.script = [[[200, 100, 300, 200]]] * 20
    detector.index = 0
    for i in range(20, 40):
        analysis = pipe.process_frame(frame, now=1000.0 + i * 0.1)

    assert analysis.people[0].identity.employee_id == "EMP002"
    # Track Budi tetap ada dan riwayatnya utuh, bukan pindah ke Ani.
    assert "emp:EMP001" in pipe._tracks         # noqa: SLF001
    assert pipe._tracks["emp:EMP001"] is budi   # noqa: SLF001
    assert pipe._tracks["emp:EMP002"] is not budi   # noqa: SLF001


def test_unknown_person_still_gets_tracked(rig):
    pipe, detector, _, _, _, _, frame = rig
    detector.script = [[[100, 100, 200, 200]]] * 10
    for i in range(10):
        analysis = pipe.process_frame(frame, now=i * 0.1)
    assert not analysis.people[0].identity.is_known
    assert pipe.num_tracked == 1


def test_history_migrates_when_person_becomes_recognized(rig):
    """Menit-menit awal sebelum dikenali tidak boleh dibuang."""
    pipe, detector, embedder, _, _, book, frame = rig
    detector.script = [[[100, 100, 200, 200]]] * 40

    for i in range(20):
        analysis = pipe.process_frame(frame, now=1000.0 + i * 0.1)
    assert not analysis.people[0].identity.is_known
    observed_before = analysis.people[0].observed_seconds

    # Karyawan didaftarkan di tengah sesi.
    book.add_employee("EMP001", "Budi")
    book.add_embedding("EMP001", embedder.vectors[1])
    for i in range(20, 40):
        analysis = pipe.process_frame(frame, now=1000.0 + i * 0.1)

    assert analysis.people[0].identity.employee_id == "EMP001"
    assert analysis.people[0].observed_seconds > observed_before
    assert pipe.num_tracked == 1


# ---------- keluaran ----------
def test_analysis_serializes_cleanly(rig):
    pipe, detector, _, _, _, _, frame = rig
    detector.script = [[[100, 100, 200, 200]]] * 3
    for i in range(3):
        analysis = pipe.process_frame(frame, now=i * 0.1)

    payload = analysis.to_dict()
    assert payload["width"] == 640 and payload["height"] == 480
    assert isinstance(payload["worst_level"], str)
    assert 0.0 <= payload["people"][0]["score"] <= 1.0
    import json
    json.dumps(payload)     # tidak boleh ada tipe yang tidak bisa di-serialize


def test_no_faces_gives_empty_analysis(rig):
    pipe, detector, _, _, _, _, frame = rig
    detector.script = [[]]
    analysis = pipe.process_frame(frame, now=0.0)
    assert analysis.people == []
    assert analysis.faces == []
    assert analysis.to_dict()["worst_level"] == "TIDAK_DIKETAHUI"


def test_max_faces_is_respected(rig):
    pipe, detector, _, _, _, _, frame = rig
    pipe.config.max_faces = 2
    detector.script = [[[0, 0, 90, 90], [100, 0, 190, 90], [200, 0, 290, 90],
                        [300, 0, 390, 90]]]
    analysis = pipe.process_frame(frame, now=0.0)
    assert len(analysis.faces) == 2


# ---------- default classifier ----------
def test_classifier_is_off_by_default():
    """CNN mati secara default.

    Terukur: pada wajah pekerja di foto lapangan nyata, 59% orang biasa
    ditandai lelah; pada wajah webcam yang diuji ia memberi 0,90-0,99 konstan
    di lima foto rentang enam bulan. Itu konstanta, bukan pengukuran.
    """
    assert PipelineConfig().enable_classifier is False


def test_pipeline_without_classifier_renormalizes_weights(rig):
    """Skor maksimum harus tetap bisa mencapai 1,0 tanpa CNN."""
    _, detector, _, _, _, book, frame = rig
    monkey = FatiguePipeline(
        config=PipelineConfig(enable_attendance=False, enable_classifier=False),
    )
    weights = monkey.fusion_config.weights.as_dict()
    assert weights["cnn"] == 0.0
    assert sum(weights.values()) == pytest.approx(1.0)
    monkey.close()


def test_explicit_classifier_is_honoured_even_when_flag_is_off(rig, monkeypatch):
    """Menyerahkan objek classifier berarti ingin ia dipakai.

    Mengabaikannya diam-diam karena sebuah flag adalah jebakan — dan itulah
    yang terjadi saat default-nya diubah jadi mati.
    """
    detector = FakeDetector()
    detector.script = [[[100, 100, 200, 200]]] * 6
    monkeypatch.setattr(pipeline_module, "FaceDetector", lambda **kw: detector)
    monkeypatch.setattr(pipeline_module, "FaceLandmarker", lambda **kw: FakeLandmarker())

    classifier = FakeClassifier()
    pipe = FatiguePipeline(
        config=PipelineConfig(enable_attendance=False, enable_classifier=False,
                              classifier_every=1),
        classifier=classifier,
    )
    assert pipe.classifier is classifier
    # Bobot CNN dipertahankan karena classifier-nya memang ada.
    assert pipe.fusion_config.weights.as_dict()["cnn"] > 0
    for i in range(3):
        pipe.process_frame(frame_of(), now=i * 0.1)
    assert classifier.calls == 3
    pipe.close()


def frame_of() -> np.ndarray:
    return np.zeros((480, 640, 3), dtype=np.uint8)
