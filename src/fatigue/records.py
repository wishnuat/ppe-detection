"""Penyimpanan riwayat fatigue — supaya kondisi kemarin masih bisa dilaporkan.

Sebelum modul ini ada, absensi tersimpan di SQLite tapi seluruh sinyal fatigue
hanya hidup di memori: PERCLOS, microsleep, dan level orang hilang begitu
aplikasi ditutup. Artinya pertanyaan paling wajar dari seorang supervisor —
"minggu lalu siapa yang paling sering mengantuk?" — tidak bisa dijawab sama
sekali.

Dua tabel, dengan peran yang sengaja berbeda:

    fatigue_samples   cuplikan berkala (default tiap 30 detik) per orang.
                      Dari sini datang rata-rata PERCLOS, lama pengamatan, dan
                      berapa lama seseorang berada di tiap level. Sampling
                      berkala dipilih daripada menyimpan tiap frame karena
                      satu shift 8 jam pada 9 fps = 260.000 baris per orang,
                      dan tidak satu pun pertanyaan laporan membutuhkan
                      resolusi sehalus itu.

    fatigue_events    hanya saat level NAIK ke WASPADA atau lebih. Ini yang
                      dibaca manusia: kapan, siapa, separah apa, dan alasannya.
                      Kalau semua perubahan level dicatat, log-nya penuh oleh
                      transisi turun yang tidak menarik siapa pun.

Keduanya berbagi file SQLite yang sama dengan `AttendanceBook`. Satu file untuk
satu titik pemasangan tetap prinsip yang sama: mudah di-backup, tidak ada
layanan tambahan yang bisa mati saat pabrik sedang berjalan.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from src.fatigue.types import FatigueLevel, PersonState

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path(os.getenv("ATTENDANCE_DB", PROJECT_ROOT / "data" / "attendance.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS fatigue_samples (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      TEXT,              -- NULL untuk orang tak dikenal
    display_name     TEXT NOT NULL,
    timestamp        REAL NOT NULL,
    level            TEXT NOT NULL,
    score            REAL NOT NULL,
    perclos          REAL NOT NULL,
    blink_rate       REAL NOT NULL,
    yawn_rate        REAL NOT NULL,
    nod_rate         REAL NOT NULL,
    microsleep_count INTEGER NOT NULL,
    longest_closure  REAL NOT NULL,
    observed_seconds REAL NOT NULL,
    cnn_score        REAL NOT NULL,
    camera           TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fs_time ON fatigue_samples(timestamp);
CREATE INDEX IF NOT EXISTS idx_fs_emp ON fatigue_samples(employee_id, timestamp);

CREATE TABLE IF NOT EXISTS fatigue_events (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id      TEXT,
    display_name     TEXT NOT NULL,
    timestamp        REAL NOT NULL,
    level            TEXT NOT NULL,
    previous_level   TEXT,
    score            REAL NOT NULL,
    perclos          REAL NOT NULL,
    microsleep_count INTEGER NOT NULL,
    longest_closure  REAL NOT NULL,
    reasons          TEXT NOT NULL,
    camera           TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_fe_time ON fatigue_events(timestamp);
CREATE INDEX IF NOT EXISTS idx_fe_emp ON fatigue_events(employee_id, timestamp);
"""

# Jarak antar-cuplikan. 30 detik memberi 960 baris per orang per shift 8 jam —
# cukup halus untuk menghitung lama seseorang berada di tiap level sampai
# ketelitian setengah menit, dan cukup jarang supaya database tetap kecil.
DEFAULT_SAMPLE_INTERVAL = 30.0


@dataclass
class FatigueEvent:
    """Satu kenaikan level yang layak dibaca manusia."""

    employee_id: str | None
    display_name: str
    timestamp: float
    level: str
    previous_level: str | None
    score: float
    perclos: float
    microsleep_count: int
    longest_closure: float
    reasons: str
    camera: str = ""

    @property
    def clock(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_row(self) -> dict:
        return {
            "waktu": self.clock,
            "employee_id": self.employee_id or "",
            "nama": self.display_name,
            "level": self.level,
            "level_sebelumnya": self.previous_level or "",
            "skor": round(self.score, 3),
            "perclos": round(self.perclos, 4),
            "microsleep": self.microsleep_count,
            "terpejam_terlama_dtk": round(self.longest_closure, 1),
            "alasan": self.reasons,
            "kamera": self.camera,
        }


class FatigueLog:
    """Menulis cuplikan berkala & kejadian fatigue ke SQLite.

    Dipakai oleh CLI, UI, dan API lewat `record()` — satu panggilan per frame.
    Kelas ini sendiri yang memutuskan kapan sesuatu layak ditulis, sehingga
    pemanggilnya tidak perlu memikirkan interval maupun deteksi perubahan
    level. Itu penting karena tiga pemanggil yang berbeda-beda kalau tidak
    akan menghasilkan kebijakan pencatatan yang berbeda-beda pula.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        sample_interval: float = DEFAULT_SAMPLE_INTERVAL,
        camera: str = "",
    ) -> None:
        self.db_path = Path(db_path or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.sample_interval = sample_interval
        self.camera = camera
        self._last_sample: dict[str, float] = {}
        self._last_level: dict[str, FatigueLevel] = {}
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @staticmethod
    def _key(person: PersonState) -> str:
        return person.identity.employee_id or f"anon:{person.identity.name}"

    # ---------- penulisan ----------
    def record(self, people: list[PersonState], now: float | None = None) -> list[FatigueEvent]:
        """Catat keadaan sekarang. Kembalikan kejadian baru yang layak dialarmkan.

        Aman dipanggil tiap frame: cuplikan hanya ditulis setiap
        `sample_interval` detik per orang, dan kejadian hanya saat level benar-
        benar naik.

        Orang dengan level TIDAK_DIKETAHUI tidak dicuplik — belum ada yang bisa
        dilaporkan tentang mereka, dan memasukkannya ke rata-rata PERCLOS akan
        mengencerkan angka orang yang benar-benar terukur.
        """
        t = time.time() if now is None else now
        events: list[FatigueEvent] = []
        samples = []

        for person in people:
            if person.level is FatigueLevel.UNKNOWN:
                continue
            key = self._key(person)

            previous = self._last_level.get(key)
            self._last_level[key] = person.level
            # Hanya kenaikan ke WASPADA ke atas yang dicatat sebagai kejadian.
            # Transisi turun tidak menarik siapa pun dan cuma menggandakan log.
            if (person.level.severity >= FatigueLevel.MILD.severity
                    and (previous is None or person.level.severity > previous.severity)):
                events.append(FatigueEvent(
                    employee_id=person.identity.employee_id,
                    display_name=person.identity.name,
                    timestamp=t,
                    level=person.level.value,
                    previous_level=previous.value if previous else None,
                    score=person.score,
                    perclos=person.perclos,
                    microsleep_count=person.microsleep_count,
                    longest_closure=person.longest_closure,
                    reasons="; ".join(person.reasons),
                    camera=self.camera,
                ))

            last = self._last_sample.get(key)
            if last is None or t - last >= self.sample_interval:
                self._last_sample[key] = t
                samples.append((
                    person.identity.employee_id, person.identity.name, t,
                    person.level.value, person.score, person.perclos,
                    person.blink_rate, person.yawn_rate, person.nod_rate,
                    person.microsleep_count, person.longest_closure,
                    person.observed_seconds, person.cnn_score, self.camera,
                ))

        if samples or events:
            with self._conn() as conn:
                if samples:
                    conn.executemany(
                        "INSERT INTO fatigue_samples (employee_id, display_name, "
                        "timestamp, level, score, perclos, blink_rate, yawn_rate, "
                        "nod_rate, microsleep_count, longest_closure, "
                        "observed_seconds, cnn_score, camera) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        samples,
                    )
                for event in events:
                    conn.execute(
                        "INSERT INTO fatigue_events (employee_id, display_name, "
                        "timestamp, level, previous_level, score, perclos, "
                        "microsleep_count, longest_closure, reasons, camera) "
                        "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (event.employee_id, event.display_name, event.timestamp,
                         event.level, event.previous_level, event.score,
                         event.perclos, event.microsleep_count,
                         event.longest_closure, event.reasons, event.camera),
                    )
        return events

    def reset_session(self) -> None:
        """Lupakan state antar-sesi (ganti shift, restart kamera).

        Tanpa ini, orang yang sudah LELAH di sesi sebelumnya tidak akan
        menghasilkan kejadian baru saat sesi berikutnya dimulai — karena
        levelnya "tidak naik" menurut ingatan lama.
        """
        self._last_sample.clear()
        self._last_level.clear()

    # ---------- pembacaan ----------
    def events(self, since: float | None = None, until: float | None = None,
               employee_id: str | None = None, limit: int = 5000) -> list[FatigueEvent]:
        query = "SELECT * FROM fatigue_events WHERE 1=1"
        params: list = []
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        if until is not None:
            query += " AND timestamp < ?"
            params.append(until)
        if employee_id is not None:
            query += " AND employee_id = ?"
            params.append(employee_id)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            FatigueEvent(
                employee_id=r["employee_id"], display_name=r["display_name"],
                timestamp=r["timestamp"], level=r["level"],
                previous_level=r["previous_level"], score=r["score"],
                perclos=r["perclos"], microsleep_count=r["microsleep_count"],
                longest_closure=r["longest_closure"], reasons=r["reasons"],
                camera=r["camera"],
            )
            for r in rows
        ]

    def samples(self, since: float | None = None, until: float | None = None,
                employee_id: str | None = None) -> list[dict]:
        query = "SELECT * FROM fatigue_samples WHERE 1=1"
        params: list = []
        if since is not None:
            query += " AND timestamp >= ?"
            params.append(since)
        if until is not None:
            query += " AND timestamp < ?"
            params.append(until)
        if employee_id is not None:
            query += " AND employee_id = ?"
            params.append(employee_id)
        query += " ORDER BY timestamp"

        with self._conn() as conn:
            return [dict(r) for r in conn.execute(query, params).fetchall()]

    def stats(self) -> dict:
        with self._conn() as conn:
            samples = conn.execute("SELECT COUNT(*) FROM fatigue_samples").fetchone()[0]
            events = conn.execute("SELECT COUNT(*) FROM fatigue_events").fetchone()[0]
            span = conn.execute(
                "SELECT MIN(timestamp), MAX(timestamp) FROM fatigue_samples"
            ).fetchone()
        return {
            "samples": samples,
            "events": events,
            "sample_interval": self.sample_interval,
            "first_sample": span[0],
            "last_sample": span[1],
        }
