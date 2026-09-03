"""Menggabungkan bukti CNN + perilaku temporal jadi satu level kelelahan.

Kenapa perlu fusi, bukan salah satu saja:

    CNN sendirian  melihat penampakan wajah (mata sayu, lingkar hitam, ekspresi
                   lemas) dan tidak melihat waktu sama sekali. Ia tidak bisa
                   membedakan orang yang berkedip dari orang yang tertidur, dan
                   ia mudah tertipu pencahayaan dan ekspresi wajah santai.
    Temporal sendirian  mengukur perilaku mata dan mulut dengan andal, tapi
                   buta terhadap tanda kelelahan yang tidak bergerak. Orang
                   yang memaksa matanya tetap terbuka punya PERCLOS rendah dan
                   tetap saja tidak layak mengoperasikan alat berat.

Keduanya salah dengan cara yang berbeda, jadi menggabungkannya menutup lubang
masing-masing.

Struktur keputusan sengaja dibuat dua lapis, bukan satu skor tunggal:

    Skor tertimbang    kombinasi lunak semua bukti -> level dasar. Cocok untuk
                       kelelahan yang menumpuk perlahan.
    Aturan keras       satu kejadian yang sudah cukup bukti dengan sendirinya
                       (microsleep, mata terpejam lama) langsung mengangkat
                       level, berapa pun skor lunaknya. Menunggu rata-rata
                       jendela 60 detik naik untuk kasus seperti itu berarti
                       peringatan datang puluhan detik terlambat.

Bobotnya bisa dikonfigurasi, dan nilai defaultnya berpihak pada PERCLOS karena
di antara semua sinyal visual, PERCLOS yang paling kuat korelasinya dengan
penurunan performa dalam literatur mengantuk.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.fatigue.temporal import TemporalSummary
from src.fatigue.types import FatigueLevel

# Titik jenuh tiap sinyal: nilai yang dianggap "bukti penuh" (kontribusi 1.0).
# Di atas itu tidak menambah apa-apa — supaya satu sinyal ekstrem tidak bisa
# mendominasi seluruh keputusan sendirian.
PERCLOS_FULL = 0.40      # PERCLOS 40% sudah masuk kategori mengantuk berat
BLINK_RATE_FULL = 32.0   # kedip/menit; normal 12-20
YAWN_RATE_FULL = 4.0     # menguap/menit
NOD_RATE_FULL = 5.0      # terkulai/menit


@dataclass
class FusionWeights:
    """Bobot relatif tiap sumber bukti. Dinormalisasi saat dipakai.

    Nilai default memenuhi satu invarian yang disengaja: **tidak ada sumber
    selain PERCLOS yang bisa menaikkan level sendirian.** Bobot CNN (0,20),
    menguap (0,20), kedip (0,10), dan terkulai (0,10) semuanya di bawah
    `mild_at` (0,30), jadi masing-masing butuh dukungan sumber lain sebelum
    apa pun dilaporkan.

    Ini bukan penyetelan yang kebetulan pas — ia menutup masalah nyata.
    Classifier menghasilkan distribusi yang hampir biner (56,7% keluarannya
    di bawah 0,05 atau di atas 0,95 pada test set), sehingga kontribusinya
    praktis berupa saklar: nol, atau bobot penuh. Dengan bobot 0,30 dan ambang
    WASPADA 0,30, satu keluaran CNN yang keliru-tapi-yakin cukup untuk membuat
    orang yang matanya terbuka lebar sepanjang menit itu dilaporkan waspada.

    PERCLOS sengaja dikecualikan: ia satu-satunya sumber yang boleh berbicara
    sendirian, karena ia pengukuran fisik langsung dengan definisi yang jelas,
    bukan tebakan model. PERCLOS 40% memang layak jadi peringatan tanpa perlu
    dikuatkan apa pun.

    `single_source_ceiling()` memeriksa invarian ini untuk bobot apa pun,
    termasuk yang diatur ulang operator lewat UI.
    """

    cnn: float = 0.20
    perclos: float = 0.40
    blink: float = 0.10
    yawn: float = 0.20
    nod: float = 0.10

    def normalized(self) -> "FusionWeights":
        total = self.cnn + self.perclos + self.blink + self.yawn + self.nod
        if total <= 0:
            raise ValueError("Total bobot fusi harus > 0")
        return FusionWeights(
            cnn=self.cnn / total, perclos=self.perclos / total,
            blink=self.blink / total, yawn=self.yawn / total, nod=self.nod / total,
        )

    def as_dict(self) -> dict[str, float]:
        w = self.normalized()
        return {"cnn": w.cnn, "perclos": w.perclos, "blink": w.blink,
                "yawn": w.yawn, "nod": w.nod}

    def without(self, *sources: str) -> "FusionWeights":
        """Bobot tanpa sumber tertentu, dinormalisasi ulang atas sisanya.

        Dipakai saat sebuah sumber benar-benar mati — misalnya classifier CNN
        yang dinonaktifkan. Kalau bobotnya sekadar dibiarkan menyumbang nol,
        skala skornya menyusut diam-diam: dengan CNN mati dan bobotnya 0,20,
        skor maksimum yang mungkin cuma 0,80, sehingga ambang KRITIS 0,70
        praktis mustahil dicapai dan seluruh sistem jadi lebih tumpul tanpa
        ada yang menyadarinya.

        Menormalisasi ulang membuat ambang tetap berarti sama.
        """
        unknown = set(sources) - set(self.as_dict())
        if unknown:
            raise ValueError(f"Sumber tidak dikenal: {sorted(unknown)}")
        kept = {k: v for k, v in vars(self).items() if k not in sources}
        if not kept or sum(kept.values()) <= 0:
            raise ValueError("Tidak boleh mematikan semua sumber bukti")
        return FusionWeights(**{k: (0.0 if k in sources else v)
                                for k, v in vars(self).items()}).normalized()


@dataclass
class FusionConfig:
    """Ambang & kebijakan yang mengubah skor jadi level."""

    weights: FusionWeights = field(default_factory=FusionWeights)
    # Batas skor untuk tiap level. Wajib menaik.
    mild_at: float = 0.30
    severe_at: float = 0.50
    critical_at: float = 0.70
    # Aturan keras.
    microsleep_level: FatigueLevel = FatigueLevel.SEVERE
    critical_closure_seconds: float = 3.0
    # Histeresis: level hanya boleh TURUN setelah bertahan sekian detik di
    # kondisi yang lebih baik. Naik selalu langsung.
    downgrade_dwell_seconds: float = 20.0

    def __post_init__(self) -> None:
        if not (self.mild_at < self.severe_at < self.critical_at):
            raise ValueError(
                "Ambang harus menaik: mild_at < severe_at < critical_at, "
                f"dapat {self.mild_at}/{self.severe_at}/{self.critical_at}"
            )

    def single_source_ceiling(self) -> dict[str, FatigueLevel]:
        """Level tertinggi yang bisa dicapai tiap sumber SENDIRIAN.

        Berguna untuk dua hal: menguji invarian "hanya PERCLOS yang boleh
        menaikkan level sendirian" di test, dan menampilkannya di UI supaya
        operator yang mengubah bobot bisa melihat konsekuensinya alih-alih
        menemukannya lewat alarm palsu seminggu kemudian.
        """
        weights = self.weights.normalized()
        ceiling = {}
        for source, weight in weights.as_dict().items():
            # Tiap sinyal jenuh di 1.0 setelah dinormalisasi, jadi kontribusi
            # maksimum satu sumber persis sama dengan bobotnya.
            if weight >= self.critical_at:
                level = FatigueLevel.CRITICAL
            elif weight >= self.severe_at:
                level = FatigueLevel.SEVERE
            elif weight >= self.mild_at:
                level = FatigueLevel.MILD
            else:
                level = FatigueLevel.ALERT
            ceiling[source] = level
        return ceiling

    def sources_that_can_escalate_alone(self) -> list[str]:
        """Sumber yang sendirian sudah cukup menaikkan level di atas SEGAR."""
        return sorted(
            source for source, level in self.single_source_ceiling().items()
            if level.severity > FatigueLevel.ALERT.severity
        )


@dataclass
class FusionResult:
    level: FatigueLevel
    score: float
    reasons: list[str] = field(default_factory=list)
    contributions: dict[str, float] = field(default_factory=dict)


def _saturate(value: float, full: float) -> float:
    """Skala linier ke 0..1, dipotong di `full`."""
    if full <= 0:
        return 0.0
    return max(0.0, min(1.0, value / full))


def _blink_evidence(rate: float) -> float:
    """Laju kedip -> bukti 0..1, dengan perlakuan khusus untuk laju rendah.

    Hubungannya tidak monoton: laju kedip naik pada kelelahan ringan lalu
    turun lagi pada kelelahan berat karena kedipan berubah jadi lebih lama
    alih-alih lebih sering. Laju di bawah 6/menit karena itu diperlakukan
    sebagai bukti lemah, bukan sebagai tanda kesegaran. Yang membedakan
    "jarang berkedip karena fokus" dari "jarang berkedip karena hampir
    tertidur" adalah PERCLOS, dan itu ditimbang terpisah.
    """
    if rate <= 0.0:
        return 0.0
    if rate < 6.0:
        return 0.35
    return _saturate(rate - 12.0, BLINK_RATE_FULL - 12.0)


class FatigueFusion:
    """Mengubah (skor CNN + ringkasan temporal) jadi level, dengan histeresis."""

    def __init__(self, config: FusionConfig | None = None) -> None:
        self.config = config or FusionConfig()
        self._weights = self.config.weights.normalized()
        # State histeresis per pemanggil; satu instance dipakai satu orang.
        self._level = FatigueLevel.UNKNOWN
        self._pending_level: FatigueLevel | None = None
        self._pending_since: float = 0.0

    @property
    def level(self) -> FatigueLevel:
        return self._level

    # ---------- skor ----------
    def score(self, summary: TemporalSummary) -> tuple[float, dict[str, float]]:
        """Skor fusi lunak 0..1 + kontribusi tiap sinyal (sudah dibobot)."""
        w = self._weights
        parts = {
            "cnn": w.cnn * max(0.0, min(1.0, summary.cnn_score)),
            "perclos": w.perclos * _saturate(summary.perclos, PERCLOS_FULL),
            "blink": w.blink * _blink_evidence(summary.blink_rate),
            "yawn": w.yawn * _saturate(summary.yawn_rate, YAWN_RATE_FULL),
            "nod": w.nod * _saturate(summary.nod_rate, NOD_RATE_FULL),
        }
        # Dijepit ke [0, 1]: bobot dinormalisasi dan tiap sinyal sudah jenuh di
        # 1.0, jadi totalnya secara matematis tidak bisa lewat 1 — tapi
        # penjumlahan float bisa mendarat di 1.0000000000000002, dan angka itu
        # bocor ke JSON API sebagai skor yang mustahil.
        total = sum(parts.values())
        return max(0.0, min(1.0, total)), {k: round(v, 4) for k, v in parts.items()}

    def _base_level(self, score: float) -> FatigueLevel:
        cfg = self.config
        if score >= cfg.critical_at:
            return FatigueLevel.CRITICAL
        if score >= cfg.severe_at:
            return FatigueLevel.SEVERE
        if score >= cfg.mild_at:
            return FatigueLevel.MILD
        return FatigueLevel.ALERT

    def _reasons(self, summary: TemporalSummary) -> list[str]:
        """Penjelasan singkat berbahasa manusia — operator harus tahu KENAPA.

        Alert tanpa alasan akan diabaikan atau dimatikan; ini bagian dari
        sistemnya, bukan hiasan.
        """
        out = []
        if summary.perclos >= 0.15:
            out.append(f"PERCLOS {summary.perclos * 100:.0f}% "
                       f"(mata tertutup {summary.perclos * 100:.0f}% waktu)")
        if summary.microsleep_count:
            out.append(f"{summary.microsleep_count}x microsleep "
                       f"(terlama {summary.longest_closure:.1f} dtk)")
        elif summary.longest_closure >= 1.0:
            out.append(f"mata terpejam {summary.longest_closure:.1f} dtk")
        if summary.yawn_rate >= 1.0:
            out.append(f"menguap {summary.yawn_rate:.1f}x/menit")
        if summary.nod_rate >= 1.0:
            out.append(f"kepala terkulai {summary.nod_rate:.1f}x/menit")
        if summary.blink_rate >= 24.0:
            out.append(f"kedip cepat {summary.blink_rate:.0f}x/menit")
        elif 0 < summary.blink_rate < 6.0:
            out.append(f"kedip sangat jarang {summary.blink_rate:.0f}x/menit")
        if summary.cnn_score >= 0.6:
            out.append(f"tampilan wajah lelah (model {summary.cnn_score:.2f})")
        return out

    # ---------- keputusan ----------
    def update(self, summary: TemporalSummary, now: float) -> FusionResult:
        """Hitung level baru dari satu ringkasan jendela.

        `now` diminta eksplisit (bukan time.time() internal) supaya histeresis
        bisa diuji deterministik dan supaya pemutaran ulang rekaman video
        memakai waktu videonya, bukan waktu dinding.
        """
        score, contributions = self.score(summary)

        if not summary.reliable:
            # Belum cukup bukti. Jangan menebak — dan jangan pula membiarkan
            # level lama menempel: yang jujur adalah menyatakan tidak tahu.
            self._level = FatigueLevel.UNKNOWN
            self._pending_level = None
            return FusionResult(
                level=FatigueLevel.UNKNOWN, score=score,
                reasons=["data belum cukup (wajah belum cukup lama terlihat jelas)"],
                contributions=contributions,
            )

        target = self._base_level(score)
        reasons = self._reasons(summary)

        # --- aturan keras ---
        cfg = self.config
        if summary.longest_closure >= cfg.critical_closure_seconds:
            if target.severity < FatigueLevel.CRITICAL.severity:
                target = FatigueLevel.CRITICAL
                # Versi ringan dari alasan yang sama sudah ditambahkan
                # `_reasons`; dibuang supaya operator tidak membaca kalimat
                # kembar yang menyebut angka yang persis sama.
                reasons = [r for r in reasons if not r.startswith("mata terpejam")]
                reasons.insert(0, f"mata terpejam {summary.longest_closure:.1f} dtk "
                                  f"(>= {cfg.critical_closure_seconds:.0f} dtk)")
        elif summary.microsleep_count > 0:
            if target.severity < cfg.microsleep_level.severity:
                target = cfg.microsleep_level
                reasons.insert(0, "terdeteksi microsleep")

        # --- histeresis ---
        # Naik langsung: kalau kondisinya memburuk, menunda peringatan adalah
        # kesalahan yang biayanya ditanggung orang di lapangan. Turun ditahan:
        # level yang berkedip-kedip antara LELAH dan SEGAR membuat operator
        # berhenti mempercayai dashboard-nya.
        if self._level is FatigueLevel.UNKNOWN or target.severity >= self._level.severity:
            self._level = target
            self._pending_level = None
        else:
            if self._pending_level is not target:
                self._pending_level = target
                self._pending_since = now
            elif now - self._pending_since >= cfg.downgrade_dwell_seconds:
                self._level = target
                self._pending_level = None

        if not reasons:
            reasons.append("tidak ada tanda kelelahan menonjol")
        return FusionResult(
            level=self._level, score=score, reasons=reasons,
            contributions=contributions,
        )

    def reset(self) -> None:
        self._level = FatigueLevel.UNKNOWN
        self._pending_level = None
        self._pending_since = 0.0
