"""CLI entrypoint for local inference (image / video / webcam)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.backends import BACKEND_LABELS, BACKENDS, build_detector as _build
from src.backends import describe, normalize
from src.detector import SELECTABLE_CATEGORIES, PPEDetector


def build_detector(args: argparse.Namespace) -> PPEDetector:
    """Pilih backend inference sesuai flag `--backend`."""
    name = normalize(args.backend)
    det = _build(
        name,
        conf=getattr(args, "conf", None),
        device=getattr(args, "device", None),
    )
    print(f"[INFO] Backend: {BACKEND_LABELS[name]} — {describe(det)}")
    return det


def apply_category_filter(detector: PPEDetector, spec: str | None) -> None:
    if not spec:
        return
    wanted = {c.strip() for c in spec.split(",") if c.strip()}
    unknown = wanted - set(SELECTABLE_CATEGORIES)
    if unknown:
        raise SystemExit(
            f"[ERROR] Kategori tidak dikenal: {', '.join(sorted(unknown))}. "
            f"Pilihan: {', '.join(SELECTABLE_CATEGORIES)}"
        )
    detector.enabled_categories = wanted
    print(f"[INFO] Kategori aktif: {', '.join(sorted(wanted))}")


def main() -> None:
    ap = argparse.ArgumentParser(description="PPE Detection CLI")
    ap.add_argument(
        "--backend",
        choices=[*BACKENDS, "local"],
        default="torch",
        help="torch = models/best.pt (PyTorch). openvino / openvino-int8 = "
             "IR terkompilasi, jauh lebih cepat di CPU & iGPU Intel. "
             "roboflow = serverless API (butuh internet). "
             "('local' = alias lama untuk torch)",
    )
    ap.add_argument(
        "--device",
        type=str,
        default=None,
        help="Khusus backend OpenVINO: CPU (default), GPU (iGPU Intel), atau AUTO.",
    )
    ap.add_argument("--conf", type=float, default=None, help="Confidence threshold (0-1)")
    ap.add_argument(
        "--categories",
        type=str,
        default=None,
        help="Batasi deteksi ke kategori tertentu, dipisah koma. "
             f"Pilihan: {', '.join(SELECTABLE_CATEGORIES)}. "
             "Contoh: --categories helmet,vest,mask",
    )

    sub = ap.add_subparsers(dest="mode", required=True)

    img = sub.add_parser("image", help="Deteksi pada satu file gambar")
    img.add_argument("path", type=str)
    img.add_argument("--out", type=str, default=None)

    vid = sub.add_parser("video", help="Deteksi pada file video")
    vid.add_argument("path", type=str)
    vid.add_argument("--out", type=str, default=None)

    cam = sub.add_parser("webcam", help="Deteksi realtime dari webcam")
    cam.add_argument("--index", type=int, default=0)
    cam.add_argument("--save", type=str, default=None,
                     help="Simpan rekaman teranotasi ke file mp4")

    args = ap.parse_args()
    detector = build_detector(args)
    apply_category_filter(detector, args.categories)

    if args.mode == "image":
        annotated, result = detector.predict_image(args.path)
        out_path = args.out or f"outputs/annotated_{Path(args.path).stem}.jpg"
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(out_path, annotated)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
        print(f"[OK] Hasil disimpan ke {out_path}")
    elif args.mode == "video":
        out = detector.predict_video(args.path, args.out)
        print(f"[OK] Video hasil deteksi: {out}")
    elif args.mode == "webcam":
        detector.predict_webcam(args.index, save_video=args.save)


if __name__ == "__main__":
    main()
