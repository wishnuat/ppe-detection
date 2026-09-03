"""Membuka webcam dengan aman di Windows — dipakai modul APD maupun fatigue.

Kenapa ini perlu ada, dan tidak cukup memanggil `cv2.VideoCapture` langsung:

`cv2.VideoCapture(index, cv2.CAP_DSHOW)` bisa **menggantung tanpa batas waktu**
di Windows. Terukur di mesin uji: pembukaan pertama setelah jeda berhasil dalam
0,3 detik, lalu dua pembukaan berikutnya tidak pernah kembali sama sekali
(dihentikan setelah 25 detik). Penyebabnya device DirectShow yang belum
dilepas oleh proses sebelumnya — dan itu terjadi setiap kali aplikasi
di-restart cepat, yang justru pola paling normal saat orang mencoba-coba.

Gejalanya jahat karena tidak terlihat seperti error: program diam, tidak ada
window, tidak ada pesan. Pemakai menyimpulkan aplikasinya rusak.

Dua hal yang diperbaiki di sini:

1. **Media Foundation didahulukan.** Pada pengujian yang sama, MSMF membuka
   kamera dalam 0,9 detik secara konsisten — termasuk pada saat DirectShow
   sedang tersangkut. Selisih 0,6 detik saat lancar tidak sebanding dengan
   risiko menggantung selamanya.

2. **Setiap percobaan dibatasi waktu.** Pembukaan dijalankan di thread
   terpisah; kalau ia tidak kembali dalam batas waktu, thread-nya ditinggalkan
   (daemon) dan backend berikutnya dicoba. Menggantung berubah jadi pesan
   error yang bisa ditindaklanjuti.
"""
from __future__ import annotations

import threading

import cv2

# Urutan backend yang dicoba. MSMF lebih dulu karena lebih andal; DSHOW
# disimpan sebagai cadangan karena pada sebagian webcam ia yang justru jalan.
BACKENDS: tuple[tuple[int, str], ...] = (
    (cv2.CAP_MSMF, "Media Foundation"),
    (cv2.CAP_DSHOW, "DirectShow"),
    (cv2.CAP_ANY, "auto"),
)

DEFAULT_TIMEOUT = 8.0


class CameraOpenError(RuntimeError):
    """Kamera tidak bisa dibuka oleh satu pun backend."""


def _try_open(index: int, api: int, timeout: float):
    """Buka satu backend dengan batas waktu. None kalau gagal atau kehabisan waktu.

    Kalau thread-nya tersangkut, ia ditinggalkan begitu saja. Itu memang
    membocorkan satu thread, tapi ia daemon dan akan ikut mati bersama proses —
    jauh lebih baik daripada menggantungkan seluruh aplikasi. `cap` yang lahir
    belakangan dari thread tersangkut sengaja TIDAK disentuh: memanggil
    `release()` dari luar pada objek yang masih dipegang thread lain adalah
    cara yang rapi untuk membuat proses crash.
    """
    box: dict[str, cv2.VideoCapture] = {}

    def worker() -> None:
        box["cap"] = cv2.VideoCapture(index, api)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        return None

    cap = box.get("cap")
    if cap is None:
        return None
    if not cap.isOpened():
        cap.release()
        return None
    return cap


def open_camera(
    index: int = 0,
    timeout: float = DEFAULT_TIMEOUT,
    warn=print,
) -> tuple[cv2.VideoCapture, str]:
    """Buka webcam `index`. Kembalikan (capture, nama_backend).

    `warn` dipanggil untuk tiap backend yang gagal, supaya pemakai melihat
    prosesnya alih-alih menunggu dalam gelap. Lempar `CameraOpenError` kalau
    semuanya gagal, dengan pesan yang menyebut apa yang bisa dicoba.
    """
    tried = []
    for api, name in BACKENDS:
        cap = _try_open(index, api, timeout)
        if cap is not None:
            return cap, name
        tried.append(name)
        if warn:
            warn(f"[WARN] Backend {name} tidak bisa membuka kamera {index} "
                 f"dalam {timeout:.0f} detik — mencoba berikutnya…")

    raise CameraOpenError(
        f"Gagal membuka kamera index {index} lewat {', '.join(tried)}.\n"
        "  Yang bisa dicoba:\n"
        "  1. Index lain (--index 1 atau 2). Banyak laptop punya dua kamera —\n"
        "     satu RGB dan satu inframerah untuk Windows Hello. Yang inframerah\n"
        "     tidak cocok untuk sistem ini.\n"
        "  2. Tutup aplikasi lain yang memakai kamera (Zoom, Teams, Chrome,\n"
        "     aplikasi Camera bawaan Windows).\n"
        "  3. Kalau baru saja menutup aplikasi ini, tunggu ~10 detik. Windows\n"
        "     kadang belum melepas device-nya, dan pembukaan berikutnya\n"
        "     tersangkut sampai ia lepas.\n"
        "  4. Cek Settings > Privacy & security > Camera."
    )


def describe_camera(cap: cv2.VideoCapture) -> str:
    """Ringkasan resolusi & FPS untuk dicetak ke log."""
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    return f"{w}x{h}" + (f" @ {fps:.0f} fps" if fps > 0 else "")
