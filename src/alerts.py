"""Kebijakan alert & statistik sesi.

Dipisah dari `PPEDetector` karena ini urusan *kebijakan*, bukan inference:
detector menjawab "APD apa yang terlihat di frame ini", sedangkan modul ini
menjawab "kapan operator perlu diteriaki".

Kenapa tidak langsung alert tiap kali status PELANGGARAN muncul: deteksi
per-frame itu berkedip. Satu frame buram atau orang menoleh sebentar sudah
cukup membuat `head_helmet` hilang dan `head_nohelmet` muncul, sehingga alert
mentah akan spam dan operator berhenti mempercayainya. Karena itu ada:

    min_frames   pelanggaran harus bertahan N frame berturut-turut dulu
    cooldown     kategori yang sama tidak dialarmkan lagi selama X detik
    categories   hanya kategori terpilih yang memicu alert (helm wajib,
                 kacamata mungkin tidak, tergantung area kerja)
    require_person  abaikan pelanggaran kalau tidak ada orang di frame

Dipakai oleh Streamlit UI; sengaja bebas Streamlit supaya bisa dites dan
dipakai ulang oleh CLI / worker CCTV.
"""
from __future__ import annotations

import csv
import io
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable

from src.detector import DetectionResult

DEFAULT_MIN_FRAMES = 3
DEFAULT_COOLDOWN = 15.0


@dataclass
class ViolationEvent:
    """Satu alert yang benar-benar dipicu (bukan tiap frame pelanggaran)."""

    timestamp: float
    category: str
    confidence: float
    frame_index: int = 0
    snapshot: str | None = None

    @property
    def clock(self) -> str:
        return time.strftime("%H:%M:%S", time.localtime(self.timestamp))

    def to_row(self) -> dict[str, str | float | int]:
        return {
            "waktu": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.timestamp)),
            "kategori": self.category,
            "confidence": round(self.confidence, 4),
            "frame": self.frame_index,
            "snapshot": self.snapshot or "",
        }


class AlertEngine:
    """Ubah status compliance per frame jadi event alert yang layak ditampilkan."""

    def __init__(
        self,
        categories: Iterable[str] | None = None,
        min_frames: int = DEFAULT_MIN_FRAMES,
        cooldown: float = DEFAULT_COOLDOWN,
        require_person: bool = False,
        max_events: int = 500,
    ) -> None:
        # None = semua kategori memicu alert.
        self.categories = set(categories) if categories is not None else None
        self.min_frames = max(1, int(min_frames))
        self.cooldown = max(0.0, float(cooldown))
        self.require_person = require_person
        self.max_events = max_events

        self.streaks: Counter[str] = Counter()
        self.last_fired: dict[str, float] = {}
        self.active: set[str] = set()
        self.events: list[ViolationEvent] = []

    # ---------- config ----------
    def configure(
        self,
        categories: Iterable[str] | None = None,
        min_frames: int | None = None,
        cooldown: float | None = None,
        require_person: bool | None = None,
    ) -> None:
        """Terapkan perubahan setting dari UI tanpa membuang riwayat event."""
        if categories is not None:
            self.categories = set(categories)
        if min_frames is not None:
            self.min_frames = max(1, int(min_frames))
        if cooldown is not None:
            self.cooldown = max(0.0, float(cooldown))
        if require_person is not None:
            self.require_person = require_person

    def watches(self, category: str) -> bool:
        return self.categories is None or category in self.categories

    # ---------- core ----------
    def update(
        self,
        result: DetectionResult,
        now: float | None = None,
        frame_index: int = 0,
    ) -> list[ViolationEvent]:
        """Proses satu frame. Kembalikan event yang baru dipicu di frame ini."""
        now = time.time() if now is None else now

        violated = {
            cat for cat, status in result.compliance.items()
            if status == "PELANGGARAN" and self.watches(cat)
        }
        if self.require_person and not any(d.category == "person" for d in result.detections):
            violated = set()

        # Kategori yang tidak melanggar di frame ini kehilangan streak-nya —
        # syarat "N frame berturut-turut" harus benar-benar berturut-turut.
        for cat in list(self.streaks):
            if cat not in violated:
                del self.streaks[cat]

        fired: list[ViolationEvent] = []
        active: set[str] = set()
        for cat in sorted(violated):
            self.streaks[cat] += 1
            if self.streaks[cat] < self.min_frames:
                continue
            active.add(cat)
            last = self.last_fired.get(cat)
            if last is not None and now - last < self.cooldown:
                continue
            self.last_fired[cat] = now
            fired.append(
                ViolationEvent(
                    timestamp=now,
                    category=cat,
                    confidence=_best_confidence(result, cat),
                    frame_index=frame_index,
                )
            )

        self.active = active
        self.events.extend(fired)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        return fired

    def reset(self) -> None:
        self.streaks.clear()
        self.last_fired.clear()
        self.active.clear()
        self.events.clear()

    # ---------- output ----------
    def counts_per_category(self) -> dict[str, int]:
        return dict(Counter(e.category for e in self.events))

    def to_csv(self) -> str:
        return events_to_csv(self.events)


def _best_confidence(result: DetectionResult, category: str) -> float:
    """Confidence tertinggi di antara deteksi pelanggaran untuk kategori ini."""
    scores = [
        d.confidence for d in result.detections
        if d.category == category and d.is_violation
    ]
    return max(scores) if scores else 0.0


def events_to_csv(events: Iterable[ViolationEvent]) -> str:
    """Event log jadi CSV siap di-download (dibuka Excel/Sheets apa adanya)."""
    buf = io.StringIO()
    writer = csv.DictWriter(
        buf, fieldnames=["waktu", "kategori", "confidence", "frame", "snapshot"]
    )
    writer.writeheader()
    for event in events:
        writer.writerow(event.to_row())
    return buf.getvalue()


@dataclass
class SessionStats:
    """Ringkasan sesi: berapa lama jalan, seberapa patuh, apa yang paling sering."""

    frames: int = 0
    violation_frames: int = 0
    per_category_frames: Counter = field(default_factory=Counter)
    started_at: float | None = None
    last_at: float | None = None

    def update(self, result: DetectionResult, now: float | None = None) -> None:
        now = time.time() if now is None else now
        if self.started_at is None:
            self.started_at = now
        self.last_at = now

        self.frames += 1
        violated = [
            cat for cat, status in result.compliance.items() if status == "PELANGGARAN"
        ]
        if violated:
            self.violation_frames += 1
        self.per_category_frames.update(violated)

    @property
    def compliance_rate(self) -> float:
        """Persentase frame yang bebas pelanggaran (0-100)."""
        if not self.frames:
            return 100.0
        return 100.0 * (self.frames - self.violation_frames) / self.frames

    @property
    def duration(self) -> float:
        if self.started_at is None or self.last_at is None:
            return 0.0
        return max(0.0, self.last_at - self.started_at)

    def reset(self) -> None:
        self.frames = 0
        self.violation_frames = 0
        self.per_category_frames.clear()
        self.started_at = None
        self.last_at = None
