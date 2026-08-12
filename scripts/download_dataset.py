"""Download dataset PPE (format YOLOv8) dari Roboflow.

Catatan penting: Roboflow **tidak** menyediakan download file weights `.pt`
untuk model yang dilatih di platform mereka (Roboflow Train). Yang bisa
diunduh adalah datasetnya. Jadi untuk inference offline, alurnya:

    python scripts/download_dataset.py
    python scripts/train.py --epochs 50        # butuh GPU untuk dataset penuh
    # -> models/best.pt

Env var yang dipakai (lihat .env.example):
    ROBOFLOW_API_KEY, ROBOFLOW_WORKSPACE, ROBOFLOW_PROJECT, ROBOFLOW_VERSION
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> int:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    workspace = os.getenv("ROBOFLOW_WORKSPACE", "wishnus-workspace")
    project_id = os.getenv("ROBOFLOW_PROJECT", "ppe-detection-hyeuz-6cijw")
    version_num = int(os.getenv("ROBOFLOW_VERSION", "2"))

    if not api_key:
        print("[ERROR] ROBOFLOW_API_KEY belum di-set di .env", file=sys.stderr)
        return 1

    datasets_dir = PROJECT_ROOT / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    expected = datasets_dir / f"{project_id.split('-hyeuz')[0]}-{version_num}"
    if (expected / "data.yaml").exists():
        print(f"[SKIP] Dataset sudah ada di {expected}")
        return 0

    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] paket `roboflow` belum terpasang. "
              "Jalankan: pip install -r requirements.txt", file=sys.stderr)
        return 1

    print(f"[INFO] Menghubungi Roboflow: {workspace}/{project_id} v{version_num}")

    # roboflow SDK menulis hasil download relatif terhadap cwd
    os.chdir(datasets_dir)

    rf = Roboflow(api_key=api_key)
    version = rf.workspace(workspace).project(project_id).version(version_num)
    dataset = version.download("yolov8")

    print(f"[OK] Dataset tersimpan di {dataset.location}")
    print("[NEXT] python scripts/train.py --epochs 50")
    return 0


if __name__ == "__main__":
    sys.exit(main())
