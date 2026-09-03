"""Export laporan absensi & fatigue ke Excel.

    python scripts/export_report.py                       # hari ini
    python scripts/export_report.py --days 7              # 7 hari terakhir
    python scripts/export_report.py --from 2026-09-01 --to 2026-09-30
    python scripts/export_report.py --out laporan_september.xlsx

Isi file (satu sheet per bagian, semuanya bisa langsung di-pivot):

    Ringkasan          periode, jumlah data, dan cara membaca angkanya
    Rekap per orang    satu baris per karyawan untuk seluruh periode
    Harian per orang   satu baris per karyawan per hari
    Log absensi        jam kedatangan
    Kejadian fatigue   setiap kenaikan level ke WASPADA atau lebih
    Karyawan           daftar terdaftar & jumlah foto wajahnya

Laporan hanya bisa berisi apa yang pernah tercatat. Data fatigue baru masuk
database saat sesi monitoring berjalan — CLI webcam atau tab Monitor di
Streamlit. Kalau sheet-nya kosong, kemungkinan besar itu sebabnya, bukan
error.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.report import build_report, to_excel  # noqa: E402

DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs" / "laporan"


def parse_date(text: str) -> float:
    """'2026-09-01' -> unix timestamp tengah malam waktu lokal."""
    try:
        parsed = time.strptime(text, "%Y-%m-%d")
    except ValueError as exc:
        raise SystemExit(
            f"[ERROR] Tanggal '{text}' tidak dikenali. Formatnya YYYY-MM-DD, "
            "mis. 2026-09-01"
        ) from exc
    return time.mktime((parsed.tm_year, parsed.tm_mon, parsed.tm_mday,
                        0, 0, 0, 0, 0, -1))


def midnight_today() -> float:
    now = time.localtime()
    return time.mktime((now.tm_year, now.tm_mon, now.tm_mday, 0, 0, 0, 0, 0, -1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=None, help="Path database absensi.")
    ap.add_argument("--days", type=int, default=None,
                    help="N hari terakhir termasuk hari ini. --days 1 = hari ini.")
    ap.add_argument("--from", dest="date_from", type=str, default=None,
                    help="Tanggal mulai (YYYY-MM-DD), inklusif.")
    ap.add_argument("--to", dest="date_to", type=str, default=None,
                    help="Tanggal akhir (YYYY-MM-DD), inklusif.")
    ap.add_argument("--out", type=Path, default=None,
                    help=f"File tujuan. Default: {DEFAULT_OUT_DIR}/laporan_<periode>.xlsx")
    args = ap.parse_args()

    if args.days is not None and (args.date_from or args.date_to):
        raise SystemExit("[ERROR] Pakai --days ATAU --from/--to, jangan keduanya.")

    if args.date_from or args.date_to:
        start = parse_date(args.date_from) if args.date_from else midnight_today()
        # --to inklusif, jadi rentangnya sampai akhir hari itu.
        end = parse_date(args.date_to) + 86400 if args.date_to else start + 86400
    elif args.days is not None:
        if args.days < 1:
            raise SystemExit("[ERROR] --days minimal 1.")
        end = midnight_today() + 86400
        start = end - args.days * 86400
    else:
        start = midnight_today()
        end = start + 86400

    if end <= start:
        raise SystemExit("[ERROR] Rentang kosong: --to lebih awal daripada --from.")

    print(f"[INFO] Menyusun laporan {time.strftime('%Y-%m-%d', time.localtime(start))} "
          f"s/d {time.strftime('%Y-%m-%d', time.localtime(end - 1))}…")
    report = build_report(db_path=args.db, start=start, end=end)

    out = args.out
    if out is None:
        label = time.strftime("%Y%m%d", time.localtime(start))
        if end - start > 86400:
            label += "_" + time.strftime("%Y%m%d", time.localtime(end - 1))
        out = DEFAULT_OUT_DIR / f"laporan_{label}.xlsx"

    path = to_excel(report, out)

    print(f"\n[OK] {path}  ({path.stat().st_size / 1024:.0f} KB)")
    print(f"     {len(report.person_days)} orang-hari terpantau, "
          f"{len(report.attendance_rows)} kedatangan, "
          f"{len(report.event_rows)} kejadian fatigue")

    summary = report.team_summary()
    if summary:
        print(f"\n{'Nama':22s} {'Terpantau':>11s} {'PERCLOS':>8s} "
              f"{'Terburuk':>10s} {'Alarm':>6s}")
        print("-" * 62)
        for row in summary:
            print(f"{row['Nama'][:22]:22s} {row['Total terpantau']:>11s} "
                  f"{row['PERCLOS rata2'] * 100:7.1f}% {row['Level terburuk']:>10s} "
                  f"{row['Total peringatan']:6d}")
    for note in report.notes:
        print(f"\n[CATATAN] {note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
