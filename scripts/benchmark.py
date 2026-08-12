"""Ukur kecepatan inference tiap backend di mesin ini.

Yang diukur adalah latency end-to-end per frame (preprocess + inference +
NMS + mapping ke objek Detection) — bukan cuma waktu forward pass — karena
itulah yang menentukan FPS sebenarnya saat memproses stream kamera.

Contoh:
    python scripts/benchmark.py
    python scripts/benchmark.py --runs 100 --source frame.jpg
    python scripts/benchmark.py --backends torch,openvino,openvino-int8

Hasil ditulis ke docs/BENCHMARK.md supaya bisa langsung ditempel ke README.
"""
from __future__ import annotations

import argparse
import glob
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backends import build_detector  # noqa: E402

# (nama backend, device). device None = tidak relevan untuk backend itu.
DEFAULT_PLAN = [
    ("torch", None),
    ("openvino", "CPU"),
    ("openvino", "GPU"),
    ("openvino-int8", "CPU"),
    ("openvino-int8", "GPU"),
]


def load_frames(source: str | None, count: int) -> list[np.ndarray]:
    """Ambil beberapa frame berbeda, bukan satu frame diulang-ulang.

    Satu gambar yang sama akan tinggal di cache CPU dan bikin angka terlalu
    optimistis; variasi jumlah objek juga mempengaruhi biaya NMS.
    """
    if source and Path(source).is_file():
        img = cv2.imread(source)
        if img is None:
            raise SystemExit(f"[ERROR] Gagal membaca {source}")
        return [img]

    pattern = source or str(PROJECT_ROOT / "datasets" / "_subset_2500" / "valid" / "images" / "*.jpg")
    paths = sorted(glob.glob(pattern))[:count]
    if not paths:
        fallback = PROJECT_ROOT / "frame.jpg"
        if fallback.exists():
            print(f"[WARN] Tidak ada gambar di {pattern}, pakai {fallback}")
            return [cv2.imread(str(fallback))]
        raise SystemExit(f"[ERROR] Tidak ada gambar untuk benchmark di {pattern}")
    return [cv2.imread(p) for p in paths]


def bench_one(backend: str, device: str | None, frames: list[np.ndarray],
              runs: int, warmup: int) -> dict | None:
    try:
        det = build_detector(backend, device=device)
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] {backend} ({device or '-'}): {exc}")
        return None

    for i in range(warmup):
        det.predict_frame(frames[i % len(frames)])

    times: list[float] = []
    n_det = 0
    for i in range(runs):
        frame = frames[i % len(frames)]
        t0 = time.perf_counter()
        res = det.predict_frame(frame)
        times.append((time.perf_counter() - t0) * 1000.0)
        n_det += len(res.detections)

    times.sort()
    return {
        "backend": backend,
        "device": device or "CPU",
        "mean_ms": statistics.mean(times),
        "median_ms": statistics.median(times),
        "p95_ms": times[min(len(times) - 1, int(len(times) * 0.95))],
        "fps": 1000.0 / statistics.mean(times),
        "detections": n_det,
    }


def to_markdown(rows: list[dict], meta: dict) -> str:
    baseline = next((r for r in rows if r["backend"] == "torch"), None)
    lines = [
        "# Benchmark Inference — PPE Detection",
        "",
        f"- CPU: `{meta['cpu']}`",
        f"- Resolusi model: `{meta['imgsz']}`",
        f"- Frame diuji: {meta['frames']} gambar, {meta['runs']} iterasi "
        f"(+{meta['warmup']} warmup)",
        "- Latency = end-to-end per frame (preprocess + inference + NMS).",
        "",
        "| Backend | Device | Mean (ms) | Median (ms) | p95 (ms) | FPS | Speedup |",
        "|---------|--------|-----------|-------------|----------|-----|---------|",
    ]
    for r in rows:
        speed = (
            f"{baseline['mean_ms'] / r['mean_ms']:.2f}x"
            if baseline and r["mean_ms"] > 0
            else "—"
        )
        lines.append(
            f"| {r['backend']} | {r['device']} | {r['mean_ms']:.1f} | "
            f"{r['median_ms']:.1f} | {r['p95_ms']:.1f} | {r['fps']:.1f} | {speed} |"
        )
    lines += ["", "> Speedup dihitung relatif terhadap backend `torch` (PyTorch CPU).", ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Benchmark backend inference PPE Detection")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--frames", type=int, default=10, help="Jumlah gambar berbeda")
    ap.add_argument("--source", type=str, default=None,
                    help="File gambar atau glob pattern")
    ap.add_argument("--backends", type=str, default=None,
                    help="Daftar backend dipisah koma; default: semua varian")
    ap.add_argument("--out", type=Path, default=PROJECT_ROOT / "docs" / "BENCHMARK.md")
    args = ap.parse_args()

    plan = DEFAULT_PLAN
    if args.backends:
        wanted = {b.strip() for b in args.backends.split(",") if b.strip()}
        plan = [(b, d) for b, d in DEFAULT_PLAN if b in wanted]

    frames = load_frames(args.source, args.frames)
    print(f"[INFO] {len(frames)} frame · {args.runs} iterasi · {args.warmup} warmup\n")

    rows = []
    for backend, device in plan:
        label = f"{backend} @ {device or '-'}"
        print(f"[RUN ] {label} ...", flush=True)
        row = bench_one(backend, device, frames, args.runs, args.warmup)
        if row:
            rows.append(row)
            print(f"[DONE] {label}: {row['mean_ms']:.1f} ms  ({row['fps']:.1f} FPS)\n")

    if not rows:
        print("[ERROR] Tidak ada backend yang bisa diuji.", file=sys.stderr)
        return 1

    import platform

    imgsz = "n/a"
    try:
        from src.openvino_detector import OpenVINODetector

        imgsz = str(OpenVINODetector().imgsz)
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "cpu": platform.processor() or platform.machine(),
        "imgsz": imgsz,
        "frames": len(frames),
        "runs": args.runs,
        "warmup": args.warmup,
    }
    md = to_markdown(rows, meta)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[OK] Ditulis ke {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
