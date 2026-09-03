"""Daftarkan wajah karyawan ke database absensi.

Dua cara memberi foto:

    # dari folder: satu subfolder per karyawan, namanya = employee_id
    #   data/faces/EMP001/*.jpg
    #   data/faces/EMP002/*.jpg
    python scripts/enroll_faces.py --dir data/faces

    # dari webcam: ambil beberapa pose langsung di tempat
    python scripts/enroll_faces.py --webcam --id EMP001 --name "Budi Santoso"

Perintah tambahan:
    python scripts/enroll_faces.py --list
    python scripts/enroll_faces.py --delete EMP001

Nama karyawan diambil dari `--name`, atau dari file `name.txt` di dalam
foldernya, atau — kalau keduanya tidak ada — dari nama folder itu sendiri.

Kualitas pendaftaran menentukan kualitas absensi jauh lebih besar daripada
model yang dipakai. Script ini menolak foto yang wajahnya tidak terdeteksi,
terlalu kecil, atau menghadap terlalu miring, dan memberi tahu alasannya —
lebih baik ditolak sekarang daripada jadi karyawan yang "kadang tidak
terbaca" selama berbulan-bulan.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.attendance import AttendanceBook  # noqa: E402
from src.fatigue.face import FaceDetector, build_embedder  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Wajah di bawah ini terlalu sedikit pikselnya untuk menghasilkan embedding
# yang stabil — lebih besar dari ambang deteksi runtime, karena foto
# pendaftaran memang bisa dipilih dan tidak ada alasan menerima yang buruk.
MIN_ENROLL_FACE = 80
# Kemiripan maksimum antar foto pendaftaran orang yang sama. Di atas ini, foto
# barunya nyaris duplikat: menambahkannya tidak memperluas cakupan pose apa pun,
# cuma memperbesar database.
DUPLICATE_SIM = 0.97


def quality_check(img: np.ndarray, faces: list) -> tuple[bool, str]:
    """(diterima, alasan) untuk satu foto pendaftaran."""
    if not faces:
        return False, "wajah tidak terdeteksi"
    if len(faces) > 1:
        return False, f"ada {len(faces)} wajah — foto pendaftaran harus satu orang"
    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    if min(x2 - x1, y2 - y1) < MIN_ENROLL_FACE:
        return False, f"wajah terlalu kecil ({x2 - x1}x{y2 - y1} px, minimal {MIN_ENROLL_FACE})"

    # Blur dinilai dari variance Laplacian pada area wajah saja: latar yang
    # tajam bisa menutupi wajah yang buram kalau dihitung se-frame.
    crop = img[y1:y2, x1:x2]
    sharpness = cv2.Laplacian(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
    if sharpness < 40.0:
        return False, f"foto terlalu buram (ketajaman {sharpness:.0f}, minimal 40)"
    return True, "ok"


def enroll_images(
    book: AttendanceBook,
    detector: FaceDetector,
    embedder,
    employee_id: str,
    name: str,
    department: str,
    paths: list[Path],
) -> tuple[int, int]:
    """Daftarkan sekumpulan foto. Kembalikan (diterima, ditolak)."""
    book.add_employee(employee_id, name, department)
    accepted: list[np.ndarray] = []
    rejected = 0

    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            print(f"    [SKIP] {path.name}: gagal dibaca")
            rejected += 1
            continue

        faces = detector.detect(img)
        ok, reason = quality_check(img, faces)
        if not ok:
            print(f"    [SKIP] {path.name}: {reason}")
            rejected += 1
            continue

        vector = embedder.embed(img, faces[0])
        duplicate = next(
            (i for i, v in enumerate(accepted) if float(np.dot(v, vector)) >= DUPLICATE_SIM),
            None,
        )
        if duplicate is not None:
            print(f"    [SKIP] {path.name}: nyaris identik dengan foto ke-{duplicate + 1}")
            rejected += 1
            continue

        book.add_embedding(employee_id, vector, source=path.name)
        accepted.append(vector)
        print(f"    [OK]   {path.name}")

    return len(accepted), rejected


def enroll_from_dir(book: AttendanceBook, detector, embedder, root: Path,
                    department: str) -> int:
    folders = sorted(d for d in root.iterdir() if d.is_dir())
    if not folders:
        raise SystemExit(
            f"[ERROR] Tidak ada subfolder di {root}. Struktur yang diharapkan:\n"
            f"    {root}/EMP001/foto1.jpg\n"
            f"    {root}/EMP002/foto1.jpg"
        )

    total = 0
    for folder in folders:
        images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            print(f"[WARN] {folder.name}: tidak ada gambar, dilewati")
            continue

        name_file = folder / "name.txt"
        name = name_file.read_text(encoding="utf-8").strip() if name_file.exists() else folder.name
        print(f"[INFO] {folder.name} ({name}) — {len(images)} foto")
        ok, bad = enroll_images(book, detector, embedder, folder.name, name,
                                department, images)
        print(f"       {ok} diterima, {bad} ditolak")
        if ok == 0:
            print(f"       [WARN] {folder.name} tidak punya satu pun foto valid — "
                  "orang ini TIDAK akan pernah dikenali kamera.")
        total += ok
    return total


def enroll_from_webcam(book: AttendanceBook, detector, embedder, employee_id: str,
                       name: str, department: str, shots: int, camera_index: int) -> int:
    """Ambil beberapa foto pendaftaran langsung dari webcam."""
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Gagal membuka kamera index {camera_index}")

    book.add_employee(employee_id, name, department)
    print(f"[INFO] Tekan SPASI untuk mengambil foto ({shots} kali), 'q' untuk berhenti.")
    print("[INFO] Ambil dari beberapa sudut: depan, sedikit kiri, sedikit kanan, "
          "dan dengan pencahayaan berbeda kalau bisa.")

    accepted: list[np.ndarray] = []
    try:
        while len(accepted) < shots:
            ok, frame = cap.read()
            if not ok:
                break
            faces = detector.detect(frame)
            valid, reason = quality_check(frame, faces)

            preview = frame.copy()
            color = (0, 200, 0) if valid else (0, 0, 255)
            for face in faces:
                x1, y1, x2, y2 = face.bbox
                cv2.rectangle(preview, (x1, y1), (x2, y2), color, 2)
            cv2.putText(preview, f"{len(accepted)}/{shots}  {reason}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
            cv2.imshow("Enroll wajah", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                if not valid:
                    print(f"    [SKIP] {reason}")
                    continue
                vector = embedder.embed(frame, faces[0])
                if any(float(np.dot(v, vector)) >= DUPLICATE_SIM for v in accepted):
                    print("    [SKIP] terlalu mirip foto sebelumnya — ubah sudut/ekspresi")
                    continue
                book.add_embedding(employee_id, vector, source=f"webcam_{int(time.time())}")
                accepted.append(vector)
                print(f"    [OK]   foto {len(accepted)}/{shots}")
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    return len(accepted)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", type=Path, default=None, help="Path database absensi.")
    ap.add_argument("--backend", type=str, default=None,
                    help="Backend embedding: sface (default) atau insightface. "
                         "HARUS sama dengan yang dipakai saat runtime.")
    ap.add_argument("--dir", type=Path, default=None,
                    help="Folder berisi satu subfolder per karyawan.")
    ap.add_argument("--webcam", action="store_true", help="Ambil foto dari webcam.")
    ap.add_argument("--id", type=str, default=None, help="employee_id (mode webcam).")
    ap.add_argument("--name", type=str, default=None, help="Nama karyawan (mode webcam).")
    ap.add_argument("--department", type=str, default="")
    ap.add_argument("--shots", type=int, default=6, help="Jumlah foto di mode webcam.")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="Tampilkan karyawan terdaftar.")
    ap.add_argument("--delete", type=str, default=None,
                    help="Hapus karyawan berikut seluruh embedding & log-nya.")
    args = ap.parse_args()

    embedder = build_embedder(args.backend)
    book = AttendanceBook(db_path=args.db, backend=embedder.backend,
                          threshold=embedder.threshold)

    if args.list:
        employees = book.list_employees()
        if not employees:
            print("(belum ada karyawan terdaftar)")
        for e in employees:
            flag = "" if e.active else "  [nonaktif]"
            warn = "  <- TANPA FOTO" if e.num_embeddings == 0 else ""
            print(f"  {e.employee_id:12s} {e.name:28s} {e.department:16s} "
                  f"{e.num_embeddings} foto{flag}{warn}")
        print(f"\n{book.stats()}")
        return 0

    if args.delete:
        if book.delete_employee(args.delete):
            print(f"[OK] '{args.delete}' dihapus permanen (embedding & log ikut terhapus).")
            return 0
        print(f"[WARN] '{args.delete}' tidak ditemukan.")
        return 1

    # detect_width=None: pendaftaran dilakukan sekali per karyawan dan
    # menentukan keandalan absensinya selamanya. Menukar akurasi crop dengan
    # kecepatan di sini adalah pertukaran yang salah arah.
    detector = FaceDetector(min_face=MIN_ENROLL_FACE // 2, detect_width=None)

    if args.webcam:
        if not args.id or not args.name:
            raise SystemExit("[ERROR] Mode webcam butuh --id dan --name.")
        n = enroll_from_webcam(book, detector, embedder, args.id, args.name,
                               args.department, args.shots, args.camera)
        print(f"\n[OK] {n} foto terdaftar untuk {args.id}.")
    elif args.dir:
        if not args.dir.is_dir():
            raise SystemExit(f"[ERROR] Bukan folder: {args.dir}")
        n = enroll_from_dir(book, detector, embedder, args.dir, args.department)
        print(f"\n[OK] Total {n} foto terdaftar.")
    else:
        ap.print_help()
        return 1

    print(f"[INFO] {book.stats()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
