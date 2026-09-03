"""Ukur latensi tiap komponen pipeline fatigue, dan FPS ujung-ke-ujung.

    python scripts/benchmark_fatigue.py --source frame.jpg
    python scripts/benchmark_fatigue.py --source frame.jpg --faces 4

Yang diukur per komponen — deteksi wajah, embedding, landmark, CNN — bukan
cuma total, karena keputusan optimasi tergantung pada mana yang dominan.
Menaikkan `classifier_every` percuma kalau ternyata landmarker yang memakan
sebagian besar waktu.

Dua hal yang membuat angka benchmark mudah salah baca, dan ditangani di sini:

**Komponen diukur setelah semua model dimuat, bukan sebelumnya.** OpenVINO,
MediaPipe/XNNPACK, dan ONNX-runtime masing-masing membuka thread pool sendiri.
Mengukur deteksi wajah di proses yang bersih memberi angka jauh lebih kecil
daripada yang benar-benar terjadi saat empat model hidup bersamaan dan berebut
inti CPU yang sama — dan angka kecil itu tidak pernah bisa dicapai di produksi.

**Backend CNN dibandingkan pada `classifier_every=1`.** Dengan setelan default
1:5, empat dari lima frame tidak menyentuh CNN sama sekali, sehingga median
latensinya identik untuk semua backend dan perbandingannya jadi tidak berarti.
Angka throughput realistis tetap dilaporkan terpisah, dengan setelan default.

FPS di sini adalah FPS *pemrosesan*, bukan FPS kamera. CCTV 25 fps tidak
menuntut analisis 25 fps: PERCLOS dan microsleep tetap terukur benar pada 8-10
fps, dan itu justru cara paling murah memuat lebih banyak kamera di satu mesin.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.face import FaceDetector  # noqa: E402
from src.fatigue.pipeline import FatiguePipeline, PipelineConfig  # noqa: E402

TARGET_FPS_PER_CAMERA = 8.0


def timeit(fn, runs: int, warmup: int = 3) -> dict:
    """Jalankan `fn` beberapa kali, laporkan statistiknya dalam ms.

    Median dilaporkan berdampingan dengan rata-rata: di CPU yang juga
    menjalankan hal lain, satu run yang kena penjadwalan buruk bisa menggeser
    rata-rata puluhan persen sementara mediannya tidak bergerak.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000)
    return {
        "mean_ms": round(statistics.fmean(samples), 2),
        "median_ms": round(statistics.median(samples), 2),
        "p95_ms": round(sorted(samples)[max(0, int(len(samples) * 0.95) - 1)], 2),
        "min_ms": round(min(samples), 2),
        "runs": runs,
    }


def tile_faces(frame: np.ndarray, detector: FaceDetector, want: int) -> np.ndarray:
    """Susun ulang frame agar berisi setidaknya `want` wajah.

    Frame di-tile berdampingan. Ini bukan adegan CCTV sungguhan, tapi untuk
    mengukur biaya *per wajah* itu justru lebih bersih: jumlah wajahnya persis
    diketahui dan ukurannya seragam, jadi selisih antara 1 dan 4 wajah benar-
    benar mencerminkan biaya per wajah dan bukan perbedaan ukuran wajah.

    Jumlah ubin dihitung dari berapa wajah yang sudah ada di frame asal —
    men-tile gambar berisi dua wajah sebanyak empat kali menghasilkan delapan,
    bukan empat.
    """
    existing = len(detector.detect(frame))
    if existing == 0 or want <= existing:
        return frame
    tiles = int(np.ceil(want / existing))
    cols = int(np.ceil(np.sqrt(tiles)))
    rows = int(np.ceil(tiles / cols))
    return np.tile(frame, (rows, cols, 1))


def build_pipelines(backends: list[str], embedder: str, max_faces: int) -> dict:
    """Bangun satu pipeline per backend CNN. Yang gagal dilewati dengan pesan."""
    built = {}
    for backend in backends:
        try:
            pipeline = FatiguePipeline(
                config=PipelineConfig(max_faces=max_faces),
                embedder_backend=embedder,
                classifier_backend=backend,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"[SKIP] {backend}: {exc}")
            continue
        if pipeline.classifier is None:
            print(f"[SKIP] {backend}: checkpoint belum ada")
            pipeline.close()
            continue
        built[backend] = pipeline
    return built


def measure_end_to_end(pipeline: FatiguePipeline, frame: np.ndarray,
                       every: int, runs: int) -> dict:
    """Latensi process_frame dengan interval CNN tertentu.

    Waktu diberikan eksplisit dan menaik supaya jendela temporalnya masuk akal
    dan tidak semua sampel menumpuk di stempel waktu yang sama.
    """
    pipeline.config.classifier_every = every
    clock = {"t": 0.0}

    def step():
        clock["t"] += 0.1
        pipeline.process_frame(frame, now=clock["t"])

    # `runs` dibulatkan ke kelipatan `every` supaya proporsi frame ber-CNN
    # sama persis dengan yang dijanjikan setelannya.
    runs = max(every, (runs // every) * every)
    stats = timeit(step, runs, warmup=every)
    stats["classifier_every"] = every
    stats["fps"] = round(1000.0 / stats["mean_ms"], 2)
    stats["cameras_at_8fps"] = round(stats["fps"] / TARGET_FPS_PER_CAMERA, 2)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", type=Path, default=PROJECT_ROOT / "frame.jpg",
                    help="Gambar uji berisi wajah. Idealnya frame CCTV yang "
                         "representatif untuk lokasi Anda — resolusi dan jarak "
                         "wajah adalah dua hal yang paling menentukan latensi.")
    ap.add_argument("--faces", type=int, default=1,
                    help="Gandakan frame sampai berisi N wajah.")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--embedder", type=str, default="sface")
    ap.add_argument("--backend", nargs="+", default=["torch", "openvino", "openvino-int8"],
                    help="Backend CNN yang dibandingkan.")
    ap.add_argument("--classifier-every", type=int, default=5,
                    help="Interval CNN untuk pengukuran throughput realistis.")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "outputs" / "fatigue" / "benchmark.json")
    args = ap.parse_args()

    frame = cv2.imread(str(args.source))
    if frame is None:
        raise SystemExit(
            f"[ERROR] Gagal membaca {args.source}.\n"
            "        Berikan gambar berisi wajah lewat --source, mis:\n"
            "        python scripts/benchmark_fatigue.py --source foto.jpg\n"
            "        (frame.jpg di root ada di .gitignore, jadi tidak ikut "
            "ter-clone.)"
        )

    probe = FaceDetector()
    if args.faces > 1:
        frame = tile_faces(frame, probe, args.faces)
    faces = probe.detect(frame)
    if not faces:
        raise SystemExit(f"[ERROR] Tidak ada wajah di {args.source} — "
                         "benchmark butuh gambar yang berisi wajah.")
    del probe
    print(f"[INFO] {args.source.name} {frame.shape[1]}x{frame.shape[0]}, "
          f"{len(faces)} wajah terdeteksi")

    pipelines = build_pipelines(args.backend, args.embedder, max(8, len(faces)))
    if not pipelines:
        raise SystemExit("[ERROR] Tidak ada backend yang bisa diuji.")

    report: dict = {
        "source": str(args.source),
        "resolution": list(frame.shape[:2][::-1]),
        "num_faces": len(faces),
        "embedder": args.embedder,
        "components": {},
        "per_frame_all_backends": {},
        "throughput": {},
    }

    # ---- komponen, diukur dengan SEMUA model sudah hidup ----
    reference = next(iter(pipelines.values()))
    faces = reference.detector.detect(frame)
    print(f"\n--- Komponen (semua {len(pipelines)} backend termuat) ---")

    report["components"]["face_detection"] = timeit(
        lambda: reference.detector.detect(frame), args.runs
    )
    report["components"]["landmarks"] = timeit(
        lambda: [reference.landmarker.analyze(frame, f) for f in faces], args.runs
    )
    if reference.embedder is not None:
        report["components"][f"embedding_{args.embedder}"] = timeit(
            lambda: [reference.embedder.embed(frame, f) for f in faces], args.runs
        )

    crops = [reference.crop_face(frame, f) for f in faces]
    for backend, pipeline in pipelines.items():
        report["components"][f"classifier_{backend}"] = timeit(
            lambda p=pipeline: p.classifier.predict_batch(crops), args.runs
        )

    for name, stats in report["components"].items():
        print(f"  {name:28s} {stats['median_ms']:8.2f} ms (median)  "
              f"p95 {stats['p95_ms']:7.2f}")

    # ---- ujung-ke-ujung, CNN tiap frame: di sini backend baru terbandingkan ----
    print("\n--- Per frame, CNN tiap frame (perbandingan backend) ---")
    for backend, pipeline in pipelines.items():
        stats = measure_end_to_end(pipeline, frame, 1, args.runs)
        report["per_frame_all_backends"][backend] = stats
        print(f"  {backend:16s} {stats['mean_ms']:8.2f} ms  {stats['fps']:6.2f} fps")

    # ---- throughput realistis dengan subsampling default ----
    print(f"\n--- Throughput realistis (CNN 1:{args.classifier_every}) ---")
    for backend, pipeline in pipelines.items():
        stats = measure_end_to_end(pipeline, frame, args.classifier_every, args.runs)
        report["throughput"][backend] = stats
        print(f"  {backend:16s} {stats['mean_ms']:8.2f} ms  {stats['fps']:6.2f} fps  "
              f"~{stats['cameras_at_8fps']:.1f} kamera/mesin")

    for pipeline in pipelines.values():
        pipeline.close()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Laporan: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
