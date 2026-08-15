"""Smoke test frontend `web/index.html` di browser sungguhan (Chrome headless).

Berbeda dari `pytest tests`, yang menguji logika Python: skrip ini menguji hal
yang hanya bisa gagal di browser — fetch ke API, penggambaran bounding box ke
<canvas>, filter slider/kategori, dan pergantian tab. Yang diperiksa termasuk
piksel canvas: kalau box pelanggaran tidak benar-benar tergambar merah, tes
gagal, bukan sekadar "tidak ada exception".

Cara pakai (API harus sudah jalan lebih dulu):

    uvicorn app.api:app --port 8000          # terminal 1
    python tests/browser/run_selftest.py     # terminal 2

Opsi:
    --url URL     Base URL API. Default http://127.0.0.1:8000
    --image PATH  Gambar sample. Default: ambil dari datasets/ atau frame.jpg
    --chrome PATH Lokasi chrome.exe kalau tidak terdeteksi otomatis

Cara kerjanya: `selftest.html` disalin sementara ke `web/` supaya dilayani dari
origin yang sama dengan API (kalau tidak, iframe ke /index.html kena
same-origin policy dan isinya tidak bisa diperiksa). File itu dihapus lagi di
akhir, sukses maupun gagal.
"""
from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WEB_DIR = PROJECT_ROOT / "web"
SELFTEST_SRC = Path(__file__).resolve().parent / "selftest.html"

CHROME_CANDIDATES = [
    Path(os.environ.get("PROGRAMFILES", r"C:\Program Files"))
    / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"))
    / "Google/Chrome/Application/chrome.exe",
    Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
    Path("/usr/bin/google-chrome"),
    Path("/usr/bin/chromium"),
    Path("/usr/bin/chromium-browser"),
]


def find_chrome(explicit: str | None) -> str:
    if explicit:
        return explicit
    for c in CHROME_CANDIDATES:
        if c.exists():
            return str(c)
    for name in ("google-chrome", "chromium", "chrome"):
        found = shutil.which(name)
        if found:
            return found
    sys.exit("[ERROR] Chrome tidak ditemukan. Pakai --chrome <path ke chrome>.")


def find_sample(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.exists():
            sys.exit(f"[ERROR] Gambar sample tidak ada: {p}")
        return p
    for pattern in ("datasets/*/test/images/*.jpg", "datasets/*/valid/images/*.jpg"):
        hits = sorted(PROJECT_ROOT.glob(pattern))
        if hits:
            return hits[0]
    fallback = PROJECT_ROOT / "frame.jpg"
    if fallback.exists():
        return fallback
    sys.exit(
        "[ERROR] Tidak ada gambar sample. Jalankan scripts/download_dataset.py "
        "atau pakai --image <path>."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--image", default=None)
    ap.add_argument("--chrome", default=None)
    args = ap.parse_args()

    base = args.url.rstrip("/")
    chrome = find_chrome(args.chrome)
    sample = find_sample(args.image)

    try:
        with urllib.request.urlopen(f"{base}/health", timeout=10) as resp:
            if resp.status != 200:
                raise urllib.error.URLError(f"status {resp.status}")
    except Exception as exc:  # noqa: BLE001
        sys.exit(
            f"[ERROR] API di {base} tidak menjawab ({exc}).\n"
            f"        Jalankan dulu: uvicorn app.api:app --port "
            f"{base.rsplit(':', 1)[-1]}"
        )

    print(f"[INFO] Chrome  : {chrome}")
    print(f"[INFO] API     : {base}")
    print(f"[INFO] Sample  : {sample.name}")

    WEB_DIR.mkdir(exist_ok=True)
    page = WEB_DIR / "_selftest.html"
    img = WEB_DIR / "_selftest_sample.jpg"
    try:
        shutil.copyfile(SELFTEST_SRC, page)
        shutil.copyfile(sample, img)

        with tempfile.TemporaryDirectory() as profile:
            proc = subprocess.run(
                [
                    chrome,
                    "--headless=new",
                    "--disable-gpu",
                    "--no-sandbox",
                    # Timer dipercepat supaya tes tidak menunggu wall-clock.
                    # selftest.html sengaja polling lewat fetch, bukan
                    # setTimeout, karena jam virtual berhenti selama ada
                    # request tertunda — lihat komentar di file itu.
                    "--virtual-time-budget=120000",
                    f"--user-data-dir={profile}",
                    "--dump-dom",
                    f"{base}/_selftest.html",
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=300,
            )
        dom = proc.stdout
    finally:
        page.unlink(missing_ok=True)
        img.unlink(missing_ok=True)

    m = re.search(r'<pre id="out">(.*?)</pre>', dom, re.S)
    if not m:
        print("[ERROR] Tidak menemukan hasil di DOM. Chrome gagal memuat halaman?")
        print(proc.stderr[-2000:])
        return 1

    report = html.unescape(m.group(1)).strip()
    print("\n" + report + "\n")

    if "SELFTEST LULUS" not in report:
        print("[FAIL] Frontend tidak lulus smoke test.")
        return 1
    print("[OK] Frontend lulus smoke test.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
