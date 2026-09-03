"""Agregasi temporal sinyal fatigue: dari frame-per-frame jadi kondisi orang.

Ini bagian yang membuat sistemnya berguna. Classifier CNN menjawab "wajah di
frame ini terlihat lelah atau tidak" — pertanyaan yang salah. Kelelahan bukan
properti satu frame: orang segar yang kebetulan sedang berkedip terlihat
persis seperti orang yang tertidur, dan bedanya hanya terlihat dari *berapa
lama* matanya tertutup.

Karena itu semua sinyal masuk ke jendela geser dan diringkas jadi ukuran yang
memang dipakai literatur mengantuk:

    PERCLOS     fraksi waktu kelopak mata tertutup dalam satu jendela. Ukuran
                paling mapan (Wierwille, 1994) dan korelasinya dengan
                penurunan performa paling kuat di antara semua sinyal visual.
    microsleep  penutupan mata terus-menerus >= 1,5 detik. Satu kejadian saja
                sudah signifikan — beda kelas dari PERCLOS yang gradual.
    blink_rate  laju kedip per menit. Naik saat lelah ringan, lalu justru
                turun saat lelah berat (kedip jadi lebih lama, bukan lebih
                sering) — makanya ia tidak pernah dipakai sendirian.
    yawn_rate   menguap per menit, dihitung dari bukaan mulut yang bertahan
                >= 1,5 detik supaya bicara dan tertawa tidak ikut terhitung.
    nod_rate    kepala terkulai per menit (pitch melewati ambang lalu kembali).

Kalibrasi per orang penting: EAR seseorang bermata sipit saat terjaga bisa
lebih rendah daripada EAR orang bermata besar saat mengantuk, jadi ambang
absolut tunggal pasti salah untuk sebagian orang. `PersonTracker` mengumpulkan
baseline mata-terbuka dari beberapa detik pertama dan menurunkan ambangnya
sendiri dari situ.

Modul ini murni Python + numpy: tidak ada model, tidak ada I/O, jadi seluruh
logikanya bisa diuji dengan sinyal sintetis tanpa kamera.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from src.fatigue.types import FatigueSignals

# Jendela pengamatan default. 60 detik adalah kompromi standar untuk PERCLOS:
# cukup panjang agar satu-dua kedipan tidak menggerakkan angkanya, cukup
# pendek agar operator tahu kondisi *sekarang*, bukan lima menit lalu.
DEFAULT_WINDOW_SECONDS = 60.0

# Penutupan mata >= ini dihitung microsleep. 1,5 dtk jauh di atas kedipan
# normal (0,1-0,4 dtk) sehingga hampir tidak mungkin false positive.
MICROSLEEP_SECONDS = 1.5
# Mulut menganga selama ini dianggap menguap, bukan bicara.
YAWN_SECONDS = 1.5
# Kepala tertunduk selama ini dianggap terkulai, bukan melihat ke bawah.
NOD_SECONDS = 1.2

# Kalibrasi: butuh sekian sampel mata-terbuka sebelum ambang personal dipakai.
CALIBRATION_SAMPLES = 45
# Ambang personal = baseline EAR terjaga x rasio ini. 0.72 diambil dari praktik
# umum PERCLOS "P80" (kelopak menutup 80% dari bukaan penuh) yang pada rasio
# EAR jatuh di sekitar angka ini.
EAR_CLOSED_RATIO = 0.72


@dataclass
class _Sample:
    """Satu observasi berstempel waktu. Disimpan mentah supaya jendela geser
    bisa dihitung ulang dengan benar meski frame rate-nya tidak stabil."""

    t: float
    eye_closed: bool
    mouth_open: bool
    nodding: bool
    cnn_score: float
    usable: bool


@dataclass
class TemporalSummary:
    """Ringkasan satu jendela pengamatan."""

    perclos: float = 0.0
    blink_rate: float = 0.0
    yawn_rate: float = 0.0
    nod_rate: float = 0.0
    microsleep_count: int = 0
    longest_closure: float = 0.0
    cnn_score: float = 0.0
    observed_seconds: float = 0.0
    usable_ratio: float = 0.0

    @property
    def reliable(self) -> bool:
        """Cukup data untuk dipercaya?

        Dua syarat: sudah mengamati >= 5 detik, dan >= 40% frame-nya benar-benar
        menghasilkan landmark. Orang yang membelakangi kamera menghasilkan
        banyak frame tak-terpakai; melaporkan "SEGAR" untuk mereka sama
        menyesatkannya dengan melaporkan "LELAH".
        """
        return self.observed_seconds >= 5.0 and self.usable_ratio >= 0.4


class PersonTracker:
    """Riwayat sinyal satu orang + kalibrasi ambang personalnya.

    Satu instance per identitas (atau per track wajah kalau orangnya belum
    dikenali). `FatigueMonitor` yang mengurus siklus hidupnya.
    """

    def __init__(
        self,
        window_seconds: float = DEFAULT_WINDOW_SECONDS,
        microsleep_seconds: float = MICROSLEEP_SECONDS,
        yawn_seconds: float = YAWN_SECONDS,
        nod_seconds: float = NOD_SECONDS,
        nod_pitch: float = 22.0,
        calibrate: bool = True,
    ) -> None:
        self.window_seconds = window_seconds
        self.microsleep_seconds = microsleep_seconds
        self.yawn_seconds = yawn_seconds
        self.nod_seconds = nod_seconds
        self.nod_pitch = nod_pitch
        self.calibrate = calibrate

        self._samples: deque[_Sample] = deque()
        # Baseline EAR saat terjaga, dikumpulkan dari frame yang blendshape-nya
        # jelas menyatakan mata terbuka — jadi kalibrasi tidak tercemar oleh
        # frame-frame saat orangnya memang sedang memejamkan mata.
        self._open_ears: deque[float] = deque(maxlen=CALIBRATION_SAMPLES * 4)
        self._ear_threshold: float | None = None

        # State bukaan/penutupan yang sedang berlangsung, untuk menghitung
        # durasi kejadian yang belum selesai.
        self._closure_start: float | None = None
        self._mouth_start: float | None = None
        self._nod_start: float | None = None
        # Semua log kejadian menyimpan (waktu_mulai, durasi) supaya `_evict`
        # dan `summarize` bisa memperlakukannya seragam.
        self._blinks: deque[tuple[float, float]] = deque()
        self._yawns: deque[tuple[float, float]] = deque()
        self._nods: deque[tuple[float, float]] = deque()
        self._microsleeps: deque[tuple[float, float]] = deque()
        self.last_seen: float = 0.0

    # ---------- kalibrasi ----------
    @property
    def calibrated(self) -> bool:
        return self._ear_threshold is not None

    @property
    def ear_threshold(self) -> float | None:
        return self._ear_threshold

    def _update_calibration(self, signals: FatigueSignals) -> None:
        if not self.calibrate or signals.ear is None:
            return
        # Hanya frame yang jelas mata-terbuka yang boleh jadi baseline.
        clearly_open = (
            signals.blink_score is not None and signals.blink_score < 0.15
        ) or (signals.blink_score is None and signals.ear > 0.27)
        if not clearly_open:
            return
        self._open_ears.append(signals.ear)
        if len(self._open_ears) >= CALIBRATION_SAMPLES:
            # Median, bukan rata-rata: satu frame buram dengan EAR ekstrem
            # tidak boleh menggeser ambang seseorang untuk selamanya.
            baseline = float(np.median(self._open_ears))
            self._ear_threshold = baseline * EAR_CLOSED_RATIO

    def _is_closed(self, signals: FatigueSignals) -> bool:
        """Keputusan mata-tertutup, memakai ambang personal kalau sudah ada."""
        if self._ear_threshold is not None and signals.ear is not None:
            geometric = signals.ear <= self._ear_threshold
            if signals.blink_score is None:
                return geometric
            # Dua sumber independen. Kalau keduanya setuju, terima. Kalau
            # tidak, blendshape masih boleh memutuskan sendiri asal ia sangat
            # yakin (>= 0.6) — ia yang sudah dinormalisasi antar-orang,
            # sedangkan EAR geometris bisa rusak hanya karena sudut kepala.
            return (geometric and signals.eye_closed) or signals.blink_score >= 0.6
        return signals.eye_closed

    # ---------- update ----------
    def update(self, signals: FatigueSignals, cnn_score: float,
               now: float | None = None) -> None:
        """Masukkan satu observasi frame."""
        t = time.time() if now is None else now
        self.last_seen = t
        self._update_calibration(signals)

        closed = self._is_closed(signals) if signals.usable else False
        nodding = (
            signals.pitch is not None and signals.pitch >= self.nod_pitch
        )

        self._samples.append(_Sample(
            t=t, eye_closed=closed, mouth_open=signals.mouth_open,
            nodding=nodding, cnn_score=float(cnn_score), usable=signals.usable,
        ))

        # Sinyal tidak terpakai (wajah menoleh/hilang) menutup kejadian yang
        # sedang berjalan alih-alih memperpanjangnya — kalau tidak, orang yang
        # berbalik badan akan tercatat "microsleep 30 detik".
        if not signals.usable:
            self._closure_start = self._mouth_start = self._nod_start = None
        else:
            self._track_event(t, closed, "_closure_start", self._blinks,
                              self.microsleep_seconds, self._microsleeps)
            self._track_event(t, signals.mouth_open, "_mouth_start", None,
                              self.yawn_seconds, self._yawns)
            self._track_event(t, nodding, "_nod_start", None,
                              self.nod_seconds, self._nods)

        self._evict(t)

    def _track_event(
        self,
        t: float,
        active: bool,
        start_attr: str,
        short_log: deque | None,
        long_seconds: float,
        long_log: deque,
    ) -> None:
        """Catat awal/akhir satu kejadian berdurasi.

        `short_log` menerima kejadian pendek (kedipan biasa), `long_log`
        menerima yang melewati `long_seconds` (microsleep / menguap / terkulai).
        Kejadian dicatat saat SELESAI, bukan saat mulai, karena durasinya baru
        diketahui pada saat itu.
        """
        start = getattr(self, start_attr)
        if active:
            if start is None:
                setattr(self, start_attr, t)
            return
        if start is None:
            return

        duration = t - start
        setattr(self, start_attr, None)
        if duration >= long_seconds:
            long_log.append((start, duration))
        elif short_log is not None and duration >= 0.05:
            # Batas bawah 0.05 dtk membuang satu frame nyasar; kedipan manusia
            # tercepat pun masih ~0.1 dtk.
            short_log.append((start, duration))

    def _evict(self, now: float) -> None:
        """Buang apa pun yang sudah keluar jendela."""
        cutoff = now - self.window_seconds
        while self._samples and self._samples[0].t < cutoff:
            self._samples.popleft()
        for log in (self._blinks, self._yawns, self._nods, self._microsleeps):
            while log and log[0][0] < cutoff:
                log.popleft()

    # ---------- ringkasan ----------
    def summarize(self, now: float | None = None) -> TemporalSummary:
        t = time.time() if now is None else now
        if not self._samples:
            return TemporalSummary()

        span = max(t - self._samples[0].t, 1e-6)
        usable = [s for s in self._samples if s.usable]
        n_usable = len(usable)

        # PERCLOS dihitung atas frame terpakai saja. Membaginya dengan total
        # frame akan mengencerkan angkanya setiap kali orangnya menoleh,
        # sehingga orang yang sering menoleh selalu tampak lebih segar.
        perclos = (sum(s.eye_closed for s in usable) / n_usable) if n_usable else 0.0
        cnn = float(np.mean([s.cnn_score for s in usable])) if n_usable else 0.0

        per_minute = 60.0 / span
        longest = max((d for _, d in self._microsleeps), default=0.0)
        if self._closure_start is not None:
            # Penutupan yang MASIH berlangsung ikut dihitung — justru kasus
            # inilah yang paling gawat dan paling perlu dilaporkan segera.
            longest = max(longest, t - self._closure_start)

        return TemporalSummary(
            perclos=perclos,
            blink_rate=len(self._blinks) * per_minute,
            yawn_rate=len(self._yawns) * per_minute,
            nod_rate=len(self._nods) * per_minute,
            microsleep_count=len(self._microsleeps),
            longest_closure=longest,
            cnn_score=cnn,
            observed_seconds=span,
            usable_ratio=n_usable / len(self._samples),
        )

    def reset(self) -> None:
        """Kosongkan riwayat tapi PERTAHANKAN kalibrasi.

        Dipakai saat orang yang sama kembali ke frame setelah pergi: riwayat
        lamanya sudah tidak relevan, tapi bentuk matanya tidak berubah dan
        mengulang kalibrasi dari nol hanya membuat sistem buta lagi selama
        beberapa detik.
        """
        self._samples.clear()
        for log in (self._blinks, self._yawns, self._nods, self._microsleeps):
            log.clear()
        self._closure_start = self._mouth_start = self._nod_start = None
