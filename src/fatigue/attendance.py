"""Absensi berbasis wajah: pendaftaran karyawan, pengenalan, dan log kehadiran.

Penyimpanan memakai SQLite satu file (`data/attendance.db`). Pilihan itu
disengaja: seluruh proyek ini dirancang jalan offline di satu mesin edge dekat
CCTV, dan menambah server database untuk data sebesar "beberapa ratus karyawan
x satu vektor 128 float" hanya menambah hal yang bisa mati saat pabrik sedang
berjalan. SQLite ikut ter-backup cukup dengan menyalin satu file.

Tiga tabel:

    employees   satu baris per karyawan (id, nama, departemen, aktif/tidak)
    embeddings  banyak baris per karyawan — SATU per foto pendaftaran
    attendance  log kehadiran

Kenapa banyak embedding per orang dan bukan satu rata-rata: satu vektor hanya
mewakili satu pose dan satu pencahayaan. Merata-ratakan foto yang beragam
justru menghasilkan vektor yang tidak mirip pose mana pun. Karena itu semua
foto pendaftaran disimpan utuh dan skor seseorang diambil dari similarity
TERTINGGI di antaranya — mendaftar 5-10 foto dari sudut dan cahaya berbeda
membuat pengenalan jauh lebih tahan terhadap kondisi lapangan.

Catatan privasi: yang disimpan adalah vektor embedding, bukan foto wajah.
Vektor itu tetap data biometrik dan tetap harus diperlakukan sebagai data
pribadi — `delete_employee` menghapusnya permanen, dan itu memang harus ada.
"""
from __future__ import annotations

import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.fatigue.types import Identity

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB = Path(os.getenv("ATTENDANCE_DB", PROJECT_ROOT / "data" / "attendance.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS employees (
    employee_id TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    department  TEXT DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS embeddings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id TEXT NOT NULL,
    backend     TEXT NOT NULL,
    dim         INTEGER NOT NULL,
    vector      BLOB NOT NULL,
    source      TEXT DEFAULT '',
    created_at  REAL NOT NULL,
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_emb_employee ON embeddings(employee_id);
-- Embedding dari backend berbeda tidak sebanding satu sama lain, jadi
-- pencarian selalu disaring per backend.
CREATE INDEX IF NOT EXISTS idx_emb_backend ON embeddings(backend);

CREATE TABLE IF NOT EXISTS attendance (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    employee_id  TEXT NOT NULL,
    event        TEXT NOT NULL,          -- 'masuk' | 'keluar'
    timestamp    REAL NOT NULL,
    similarity   REAL NOT NULL,
    camera       TEXT DEFAULT '',
    FOREIGN KEY (employee_id) REFERENCES employees(employee_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_att_employee_time ON attendance(employee_id, timestamp);
"""

# Jeda minimum antara dua pencatatan untuk orang yang sama. Tanpa ini, satu
# orang yang berdiri di depan kamera akan menghasilkan satu baris log per
# frame — ribuan baris untuk satu kedatangan.
DEFAULT_LOG_COOLDOWN = 300.0


@dataclass
class Employee:
    employee_id: str
    name: str
    department: str = ""
    active: bool = True
    num_embeddings: int = 0

    def to_dict(self) -> dict:
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "department": self.department,
            "active": self.active,
            "num_embeddings": self.num_embeddings,
        }


@dataclass
class AttendanceRecord:
    employee_id: str
    name: str
    event: str
    timestamp: float
    similarity: float
    camera: str = ""

    @property
    def clock(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp))

    def to_row(self) -> dict:
        return {
            "waktu": self.clock,
            "employee_id": self.employee_id,
            "nama": self.name,
            "event": self.event,
            "similarity": round(self.similarity, 4),
            "kamera": self.camera,
        }


class AttendanceBook:
    """Daftar karyawan + pencocokan wajah + log kehadiran.

    Embedding di-cache di memori sebagai satu matriks per backend. Pencocokan
    jadi satu perkalian matriks alih-alih query per wajah — untuk 200 karyawan
    x 8 foto itu matriks 1600x128, dan pencocokannya jatuh ke orde mikrodetik.
    Cache di-invalidate setiap kali pendaftaran berubah.
    """

    def __init__(
        self,
        db_path: str | Path | None = None,
        backend: str = "sface",
        threshold: float = 0.40,
        log_cooldown: float = DEFAULT_LOG_COOLDOWN,
    ) -> None:
        self.db_path = Path(db_path or DEFAULT_DB)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.threshold = threshold
        self.log_cooldown = log_cooldown

        self._matrix: np.ndarray | None = None
        self._owners: list[str] = []
        self._names: dict[str, str] = {}
        self._init_db()

    # ---------- infrastruktur ----------
    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # SQLite mematikan foreign key per default; tanpa ini, ON DELETE
        # CASCADE di skema tidak akan pernah jalan dan menghapus karyawan
        # meninggalkan embedding yatim yang masih ikut dicocokkan.
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    def _invalidate(self) -> None:
        self._matrix = None

    # ---------- pendaftaran ----------
    def add_employee(self, employee_id: str, name: str, department: str = "") -> Employee:
        """Daftarkan karyawan (idempoten: memanggil ulang memperbarui nama)."""
        if not employee_id or not name:
            raise ValueError("employee_id dan name wajib diisi")
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO employees (employee_id, name, department, active, created_at) "
                "VALUES (?, ?, ?, 1, ?) "
                "ON CONFLICT(employee_id) DO UPDATE SET name=excluded.name, "
                "department=excluded.department",
                (employee_id, name, department, time.time()),
            )
        self._invalidate()
        return Employee(employee_id, name, department)

    def add_embedding(
        self, employee_id: str, vector: np.ndarray, source: str = ""
    ) -> None:
        """Tambahkan satu vektor wajah untuk karyawan yang sudah terdaftar."""
        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vec))
        if norm < 1e-6:
            raise ValueError(
                "Vektor embedding kosong — wajah kemungkinan gagal terdeteksi "
                "pada foto ini. Pakai foto lain."
            )
        vec = vec / norm

        with self._conn() as conn:
            exists = conn.execute(
                "SELECT 1 FROM employees WHERE employee_id = ?", (employee_id,)
            ).fetchone()
            if not exists:
                raise KeyError(
                    f"Karyawan '{employee_id}' belum terdaftar. "
                    "Panggil add_employee() dulu."
                )
            conn.execute(
                "INSERT INTO embeddings (employee_id, backend, dim, vector, source, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (employee_id, self.backend, len(vec), vec.tobytes(), source, time.time()),
            )
        self._invalidate()

    def delete_employee(self, employee_id: str) -> bool:
        """Hapus karyawan berikut seluruh embedding & log-nya. Permanen."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM employees WHERE employee_id = ?", (employee_id,)
            )
            deleted = cur.rowcount > 0
        self._invalidate()
        return deleted

    def set_active(self, employee_id: str, active: bool) -> None:
        """Nonaktifkan tanpa menghapus (karyawan cuti / pindah shift)."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE employees SET active = ? WHERE employee_id = ?",
                (1 if active else 0, employee_id),
            )
        self._invalidate()

    def list_employees(self, active_only: bool = False) -> list[Employee]:
        query = (
            "SELECT e.employee_id, e.name, e.department, e.active, "
            "       COUNT(m.id) AS n "
            "FROM employees e "
            "LEFT JOIN embeddings m ON m.employee_id = e.employee_id AND m.backend = ? "
            + ("WHERE e.active = 1 " if active_only else "")
            + "GROUP BY e.employee_id ORDER BY e.name"
        )
        with self._conn() as conn:
            rows = conn.execute(query, (self.backend,)).fetchall()
        return [
            Employee(r["employee_id"], r["name"], r["department"], bool(r["active"]), r["n"])
            for r in rows
        ]

    # ---------- pencocokan ----------
    def _load_matrix(self) -> None:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT m.employee_id, m.vector, m.dim, e.name "
                "FROM embeddings m JOIN employees e ON e.employee_id = m.employee_id "
                "WHERE m.backend = ? AND e.active = 1",
                (self.backend,),
            ).fetchall()

        if not rows:
            self._matrix = np.zeros((0, 0), dtype=np.float32)
            self._owners, self._names = [], {}
            return

        dims = {r["dim"] for r in rows}
        if len(dims) > 1:
            raise ValueError(
                f"Embedding backend '{self.backend}' punya dimensi campur {sorted(dims)}. "
                "Database ini kemungkinan diisi oleh dua versi model berbeda — "
                "daftarkan ulang, jangan dicampur."
            )
        self._matrix = np.stack(
            [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        )
        self._owners = [r["employee_id"] for r in rows]
        self._names = {r["employee_id"]: r["name"] for r in rows}

    def identify(self, vector: np.ndarray) -> Identity:
        """Cocokkan satu vektor wajah ke karyawan terdaftar.

        Skor seseorang = similarity TERTINGGI di antara foto pendaftarannya,
        bukan rata-rata: rata-rata menghukum orang yang mendaftar dengan banyak
        pose beragam, padahal justru merekalah yang paling siap dikenali.
        """
        if self._matrix is None:
            self._load_matrix()
        assert self._matrix is not None

        if self._matrix.size == 0:
            return Identity.unknown()

        vec = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vec.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                f"Dimensi vektor {vec.shape[0]} != dimensi terdaftar "
                f"{self._matrix.shape[1]}. Backend embedder berbeda dari saat "
                "pendaftaran."
            )
        sims = self._matrix @ vec
        best = int(np.argmax(sims))
        score = float(sims[best])
        if score < self.threshold:
            return Identity.unknown(similarity=score)
        emp_id = self._owners[best]
        return Identity(
            employee_id=emp_id, name=self._names[emp_id],
            similarity=score, is_known=True,
        )

    def identify_many(self, vectors: list[np.ndarray]) -> list[Identity]:
        return [self.identify(v) for v in vectors]

    # ---------- log kehadiran ----------
    def last_event(self, employee_id: str) -> AttendanceRecord | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT a.*, e.name FROM attendance a "
                "JOIN employees e ON e.employee_id = a.employee_id "
                "WHERE a.employee_id = ? ORDER BY a.timestamp DESC LIMIT 1",
                (employee_id,),
            ).fetchone()
        if row is None:
            return None
        return AttendanceRecord(
            row["employee_id"], row["name"], row["event"],
            row["timestamp"], row["similarity"], row["camera"],
        )

    def check_in(
        self,
        identity: Identity,
        camera: str = "",
        now: float | None = None,
    ) -> AttendanceRecord | None:
        """Catat kehadiran. None kalau ditolak (tidak dikenal / masih cooldown).

        Cooldown-nya per orang, bukan global: dua karyawan yang lewat
        bersamaan harus dua-duanya tercatat.
        """
        if not identity.is_known or identity.employee_id is None:
            return None
        t = time.time() if now is None else now

        last = self.last_event(identity.employee_id)
        if last is not None and t - last.timestamp < self.log_cooldown:
            return None

        record = AttendanceRecord(
            employee_id=identity.employee_id, name=identity.name,
            event="masuk", timestamp=t, similarity=identity.similarity, camera=camera,
        )
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO attendance (employee_id, event, timestamp, similarity, camera) "
                "VALUES (?, ?, ?, ?, ?)",
                (record.employee_id, record.event, record.timestamp,
                 record.similarity, record.camera),
            )
        return record

    def records(
        self,
        since: float | None = None,
        employee_id: str | None = None,
        limit: int = 500,
    ) -> list[AttendanceRecord]:
        query = (
            "SELECT a.*, e.name FROM attendance a "
            "JOIN employees e ON e.employee_id = a.employee_id WHERE 1=1"
        )
        params: list = []
        if since is not None:
            query += " AND a.timestamp >= ?"
            params.append(since)
        if employee_id is not None:
            query += " AND a.employee_id = ?"
            params.append(employee_id)
        query += " ORDER BY a.timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            AttendanceRecord(r["employee_id"], r["name"], r["event"],
                             r["timestamp"], r["similarity"], r["camera"])
            for r in rows
        ]

    def today(self) -> list[AttendanceRecord]:
        """Log sejak tengah malam waktu lokal."""
        now = time.localtime()
        midnight = time.mktime((now.tm_year, now.tm_mon, now.tm_mday,
                                0, 0, 0, 0, 0, -1))
        return self.records(since=midnight)

    def stats(self) -> dict:
        with self._conn() as conn:
            employees = conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0]
            active = conn.execute(
                "SELECT COUNT(*) FROM employees WHERE active = 1"
            ).fetchone()[0]
            embeddings = conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE backend = ?", (self.backend,)
            ).fetchone()[0]
            logs = conn.execute("SELECT COUNT(*) FROM attendance").fetchone()[0]
        return {
            "db_path": str(self.db_path),
            "backend": self.backend,
            "threshold": self.threshold,
            "employees": employees,
            "active_employees": active,
            "embeddings": embeddings,
            "attendance_records": logs,
            "present_today": len({r.employee_id for r in self.today()}),
        }
