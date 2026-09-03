"""CLI fatigue detection & absensi (gambar / video / webcam).

    python -m src.fatigue.cli webcam
    python -m src.fatigue.cli video rekaman.mp4 --out outputs/hasil.mp4
    python -m src.fatigue.cli image foto.jpg

Catatan penting soal mode `image`: satu gambar diam tidak bisa menghasilkan
penilaian kelelahan yang berarti. PERCLOS, laju kedip, dan microsleep semuanya
butuh waktu, dan dari satu frame level yang jujur adalah TIDAK_DIKETAHUI. Mode
ini karena itu hanya melaporkan wajah, identitas, dan sinyal per-frame
mentahnya — berguna untuk mengecek pendaftaran wajah dan kalibrasi kamera,
bukan untuk menilai orang.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

from src.fatigue.attendance import AttendanceBook
from src.fatigue.pipeline import FatiguePipeline, PipelineConfig
from src.fatigue.types import FatigueLevel


def build(args: argparse.Namespace) -> FatiguePipeline:
    config = PipelineConfig(
        window_seconds=args.window,
        enable_attendance=not args.no_attendance,
        enable_classifier=args.classifier,
        max_faces=args.max_faces,
        camera_name=args.camera_name,
    )
    attendance = None
    if config.enable_attendance and args.db:
        attendance = AttendanceBook(db_path=args.db)
    pipeline = FatiguePipeline(
        config=config,
        attendance=attendance,
        embedder_backend=args.embedder,
        classifier_backend=args.classifier_backend,
    )
    print("[INFO] " + json.dumps(pipeline.describe(), ensure_ascii=False))
    return pipeline


def run_image(pipeline: FatiguePipeline, path: str, out: str | None) -> None:
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"[ERROR] Gagal membaca gambar: {path}")

    analysis = pipeline.process_frame(img)
    payload = analysis.to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("\n[CATATAN] Level dari satu frame selalu TIDAK_DIKETAHUI — kelelahan "
          "hanya terbaca dari rentang waktu. Pakai mode video/webcam untuk itu.")

    out_path = out or f"outputs/fatigue_{Path(path).stem}.jpg"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, pipeline.render(img, analysis))
    print(f"[OK] Hasil: {out_path}")


def run_video(pipeline: FatiguePipeline, path: str, out: str | None,
              stride: int) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Gagal membuka video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_path = out or f"outputs/fatigue_{Path(path).stem}.mp4"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Waktu diambil dari posisi frame di dalam video, BUKAN dari jam dinding.
    # Kalau memakai jam dinding, memproses rekaman 10 menit yang butuh 30 menit
    # komputasi akan membuat semua jendela temporal salah tiga kali lipat.
    index = 0
    worst: dict[str, FatigueLevel] = {}
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = index / fps
            if index % max(1, stride) == 0:
                analysis = pipeline.process_frame(frame, now=t)
                annotated = pipeline.draw_hud(
                    pipeline.render(frame, analysis), analysis
                )
                for person in analysis.people:
                    key = person.identity.name
                    if person.level.severity > worst.get(key, FatigueLevel.UNKNOWN).severity:
                        worst[key] = person.level
            else:
                annotated = frame
            writer.write(annotated)
            index += 1
            if index % 200 == 0:
                print(f"    {index} frame ({t:.0f}s video)")
    finally:
        cap.release()
        writer.release()

    print(f"\n[OK] Video hasil: {out_path}")
    print(f"[OK] {index} frame diproses.")
    if worst:
        print("[RINGKASAN] Level terburuk per orang:")
        for name, level in sorted(worst.items(), key=lambda kv: -kv[1].severity):
            print(f"    {name:28s} {level.value}")


def run_webcam(pipeline: FatiguePipeline, index: int, save: str | None) -> None:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Gagal membuka kamera index {index}")

    writer = None
    if save:
        Path(save).parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Tekan 'q' untuk berhenti, 's' untuk snapshot.")
    print("[INFO] Beberapa detik pertama dipakai untuk kalibrasi mata — "
          "level akan TIDAK_DIKETAHUI sampai datanya cukup.")

    prev_levels: dict[str, FatigueLevel] = {}
    # `recent_checkins` hanya menahan 50 terakhir, jadi indeks ini bisa
    # tertinggal di belakang kalau sesinya sangat ramai — konsekuensinya cuma
    # beberapa baris log tidak tercetak, sedangkan datanya tetap utuh di DB.
    printed_checkins = 0
    t_prev = time.perf_counter()
    fps = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            analysis = pipeline.process_frame(frame)

            now = time.perf_counter()
            dt = now - t_prev
            t_prev = now
            if dt > 0:
                fps = 0.8 * fps + 0.2 * (1.0 / dt) if fps else 1.0 / dt

            annotated = pipeline.draw_hud(pipeline.render(frame, analysis), analysis, fps)

            # Hanya cetak saat level seseorang BERUBAH, bukan tiap frame.
            for person in analysis.people:
                key = person.identity.employee_id or person.identity.name
                if prev_levels.get(key) is not person.level:
                    prev_levels[key] = person.level
                    if person.level.severity >= FatigueLevel.MILD.severity:
                        print(f"[ALERT] {person.identity.name}: {person.level.value} "
                              f"— {'; '.join(person.reasons)}")
                    elif person.level is FatigueLevel.ALERT:
                        print(f"[OK]    {person.identity.name}: kembali segar")

            while printed_checkins < len(pipeline.recent_checkins):
                record = pipeline.recent_checkins[printed_checkins]
                printed_checkins += 1
                print(f"[ABSEN] {record.clock}  {record.employee_id}  {record.name} "
                      f"(similarity {record.similarity:.3f})")

            if save:
                if writer is None:
                    h, w = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        save, cv2.VideoWriter_fourcc(*"mp4v"), max(1.0, fps), (w, h)
                    )
                writer.write(annotated)

            cv2.imshow("Fatigue Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                snap = Path("outputs") / f"fatigue_snapshot_{int(time.time())}.jpg"
                snap.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(snap), annotated)
                print(f"[OK] Snapshot: {snap}")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
            print(f"[OK] Rekaman: {save}")
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        pipeline.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Fatigue Detection & Absensi CLI")
    ap.add_argument("--embedder", type=str, default=None,
                    help="Backend face recognition: sface (default) | insightface")
    ap.add_argument("--classifier-backend", type=str, default=None,
                    help="Backend CNN fatigue: torch (default) | openvino | openvino-int8")
    ap.add_argument("--db", type=Path, default=None, help="Path database absensi.")
    ap.add_argument("--window", type=float, default=60.0,
                    help="Panjang jendela pengamatan temporal (detik).")
    ap.add_argument("--max-faces", type=int, default=8)
    ap.add_argument("--camera-name", type=str, default="",
                    help="Label kamera yang ikut tercatat di log absensi.")
    ap.add_argument("--no-attendance", action="store_true",
                    help="Matikan absensi (tidak memuat model embedding 37 MB).")
    ap.add_argument("--classifier", action=argparse.BooleanOptionalAction,
                    default=False,
                    help="Nyalakan CNN penampakan wajah. MATI secara default: "
                         "pada wajah kamera nyata ia terukur tidak andal (59% "
                         "pekerja biasa ditandai lelah). Nyalakan setelah model "
                         "dilatih ulang dengan frame dari kamera Anda sendiri.")

    sub = ap.add_subparsers(dest="mode", required=True)

    img = sub.add_parser("image", help="Analisis satu gambar (tanpa level fatigue)")
    img.add_argument("path", type=str)
    img.add_argument("--out", type=str, default=None)

    vid = sub.add_parser("video", help="Analisis file video")
    vid.add_argument("path", type=str)
    vid.add_argument("--out", type=str, default=None)
    vid.add_argument("--stride", type=int, default=1,
                     help="Analisis tiap N frame (frame di sela disalin apa adanya).")

    cam = sub.add_parser("webcam", help="Analisis realtime dari webcam/CCTV")
    cam.add_argument("--index", type=int, default=0)
    cam.add_argument("--save", type=str, default=None)

    args = ap.parse_args()
    pipeline = build(args)

    if args.mode == "image":
        run_image(pipeline, args.path, args.out)
    elif args.mode == "video":
        run_video(pipeline, args.path, args.out, args.stride)
    elif args.mode == "webcam":
        run_webcam(pipeline, args.index, args.save)


if __name__ == "__main__":
    sys.exit(main())
