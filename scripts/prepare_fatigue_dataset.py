"""Unduh dataset fatigue Kaggle, potong wajahnya, dan pecah jadi train/val/test.

    python scripts/prepare_fatigue_dataset.py

Sumber: https://www.kaggle.com/datasets/rihabkaci99/fatigue-dataset
(2.200 gambar wajah, MIT, seimbang 1.100 Fatigue / 1.100 NonFatigue, tanpa
split bawaan). Bisa diunduh anonim lewat endpoint publik Kaggle, jadi tidak
perlu API key — kalau suatu saat Kaggle menutupnya, `--zip` menerima file yang
sudah diunduh manual.


Dua keputusan yang menentukan kualitas angka akhir
--------------------------------------------------

1. Gambar dipotong ke wajah, tidak dipakai apa adanya.
   Saat runtime, classifier hanya pernah melihat crop wajah keluaran YuNet
   (`FaceDetector` + margin 25%). Kalau ia dilatih pada foto utuh berisi bahu,
   latar, dan kadang dua orang, distribusi trainingnya beda dari distribusi
   inferensi: akurasi validasi bagus, akurasi lapangan jatuh. Memotong di sini
   membuat kedua sisi memakai preprocessing yang sama persis.

2. Split dikelompokkan per identitas wajah, bukan per gambar (`--split-mode
   identity`, default).
   Dataset ini dikumpulkan dari internet, jadi satu orang muncul di banyak
   gambar. Pada split acak murni, 32% gambar test ternyata punya foto orang
   yang sama di train (cosine SFace >= 0.40). Model bisa mendapat akurasi
   tinggi hanya dengan menghafal wajah — "si A biasanya lelah" — dan angka
   test-nya tidak berarti apa-apa untuk karyawan yang belum pernah dilihat.
   Karena itu wajah diklaster dulu dengan embedding SFace, lalu satu klaster
   utuh masuk ke satu split saja. Angkanya turun; angkanya jujur.
   `--split-mode random` disediakan untuk mereproduksi baseline yang
   menggelembung itu sebagai pembanding.

Gambar yang wajahnya tidak terdeteksi TIDAK dibuang: sebagian besar memang
sudah berupa potret rapat, jadi dipakai utuh dan dicatat di manifest supaya
proporsinya bisa diaudit.

Hasil:
    datasets/fatigue/{train,val,test}/{fatigue,nonfatigue}/*.jpg
    datasets/fatigue/manifest.json
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.fatigue.face import FaceDetector, SFaceEmbedder  # noqa: E402

KAGGLE_REF = "rihabkaci99/fatigue-dataset"
KAGGLE_URL = f"https://www.kaggle.com/api/v1/datasets/download/{KAGGLE_REF}"
DEFAULT_OUT = PROJECT_ROOT / "datasets" / "fatigue"

# Nama folder di dalam zip -> label yang dipakai proyek ini.
SOURCE_CLASSES = {"Fatigue": "fatigue", "NonFatigue": "nonfatigue"}
CLASSES = ("nonfatigue", "fatigue")   # indeks 0, 1 — urutan ini dipakai model
SPLITS = ("train", "val", "test")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Ambang klaster identitas: sengaja disamakan dengan ambang "orang yang sama"
# milik absensi (`face.DEFAULT_THRESHOLDS["sface"]`). Karena klasterisasinya
# single-linkage, menyamakan keduanya membuat nol kebocoran terjamin secara
# konstruksi, bukan kebetulan: kalau ada pasangan lintas-split dengan
# similarity >= 0.40, penutup transitif pasti sudah menyatukannya lebih dulu.
#
# Diukur pada dataset ini (lihat docs/FATIGUE.md):
#     0.36 -> bocor 0%, tapi klaster terbesar menelan 1.214 dari 2.200 gambar
#     0.40 -> bocor 0%, klaster terbesar 116  <- dipakai
#     0.45 -> klaster rapi, tapi bocor 17% ke val dan 14% ke test
# Di bawah 0.40 penutup transitif mulai menggumpal — banyak orang berbeda
# terangkai jadi satu klaster raksasa yang memaksa seperempat dataset masuk
# ke satu split saja.
IDENTITY_SIM_THRESHOLD = 0.40


# ------------------------------------------------------------------ unduhan
def download_zip(dest: Path) -> Path:
    """Unduh zip dataset kalau belum ada di `dest`."""
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"[INFO] Zip sudah ada, pakai ulang: {dest} "
              f"({dest.stat().st_size / 1e6:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    print(f"[INFO] Mengunduh {KAGGLE_REF} (~34 MB)…")
    try:
        with urllib.request.urlopen(KAGGLE_URL, timeout=300) as resp, tmp.open("wb") as out:
            shutil.copyfileobj(resp, out)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        raise SystemExit(
            f"[ERROR] Gagal mengunduh dari Kaggle: {exc}\n"
            f"        Unduh manual dari https://www.kaggle.com/datasets/{KAGGLE_REF}\n"
            f"        lalu jalankan ulang dengan --zip path/ke/archive.zip"
        ) from exc
    tmp.replace(dest)
    print(f"[OK] Tersimpan: {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    return dest


def extract(zip_path: Path, raw_dir: Path) -> dict[str, list[Path]]:
    """Ekstrak zip lalu kelompokkan file gambar per label."""
    if raw_dir.exists():
        shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)

    grouped: dict[str, list[Path]] = {label: [] for label in CLASSES}
    for src_name, label in SOURCE_CLASSES.items():
        # Struktur zip: Data/<Kelas>/*.jpg. Dicari lewat rglob supaya tetap
        # jalan kalau upstream menambah/menghapus satu level folder.
        matches = [d for d in raw_dir.rglob(src_name) if d.is_dir()]
        if not matches:
            raise SystemExit(
                f"[ERROR] Folder '{src_name}' tidak ada di dalam zip. "
                f"Isi zip: {[p.name for p in raw_dir.iterdir()]}"
            )
        for folder in matches:
            grouped[label].extend(
                p for p in sorted(folder.rglob("*"))
                if p.suffix.lower() in IMAGE_SUFFIXES
            )
    return grouped


# ------------------------------------------------------------------ crop
def crop_face(detector: FaceDetector, img: np.ndarray, margin: float):
    """(crop, FaceBox) untuk wajah terbesar. (None, None) kalau tidak ada."""
    faces = detector.detect(img)
    if not faces:
        return None, None
    face = faces[0]
    x1, y1, x2, y2 = face.bbox
    h, w = img.shape[:2]
    mw, mh = int((x2 - x1) * margin), int((y2 - y1) * margin)
    crop = img[max(0, y1 - mh):min(h, y2 + mh), max(0, x1 - mw):min(w, x2 + mw)]
    return (crop, face) if crop.size else (None, None)


# ------------------------------------------------------------------ klaster
def cluster_identities(embeddings: np.ndarray, threshold: float) -> np.ndarray:
    """Klaster wajah dengan union-find single-linkage atas cosine similarity.

    Single-linkage dipilih (bukan complete/average) karena tujuannya bukan
    klaster yang rapi, melainkan *penutup transitif*: kalau A mirip B dan B
    mirip C, ketiganya harus berakhir di split yang sama walaupun A dan C
    sendiri tidak terlalu mirip. Itu justru sifat yang kita perlukan untuk
    mencegah kebocoran.

    Baris embedding sudah ternormalisasi, jadi matriks kemiripan = E @ E.T.
    Untuk 2.200 wajah itu 4,8 juta sel — cukup di memori, tidak perlu blocking.
    """
    n = len(embeddings)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    sim = embeddings @ embeddings.T
    rows, cols = np.where(np.triu(sim, k=1) >= threshold)
    for a, b in zip(rows.tolist(), cols.tolist()):
        union(a, b)

    roots = np.array([find(i) for i in range(n)])
    # Renumbering jadi 0..k-1 supaya id klaster enak dibaca di manifest.
    _, cluster_ids = np.unique(roots, return_inverse=True)
    return cluster_ids


def assign_clusters(
    cluster_ids: np.ndarray,
    labels: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> np.ndarray:
    """Tempatkan tiap klaster utuh ke satu split, menjaga proporsi per kelas.

    Greedy: klaster besar didahulukan (paling sulit ditempatkan belakangan),
    lalu tiap klaster masuk ke split yang saat itu paling jauh dari kuotanya.
    Defisit dihitung per kelas, jadi split tidak cuma seimbang jumlahnya tapi
    juga komposisi fatigue/non-fatigue-nya.
    """
    n = len(cluster_ids)
    train_frac = 1.0 - val_frac - test_frac
    targets = {"train": train_frac, "val": val_frac, "test": test_frac}

    members: dict[int, list[int]] = defaultdict(list)
    for i, c in enumerate(cluster_ids.tolist()):
        members[c].append(i)

    class_totals = np.bincount(labels, minlength=len(CLASSES)).astype(float)
    quota = {
        s: class_totals * frac for s, frac in targets.items()
    }
    current = {s: np.zeros(len(CLASSES)) for s in SPLITS}

    rng = random.Random(seed)
    order = sorted(members.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    # Acak di dalam kelompok berukuran sama supaya hasilnya tidak bergantung
    # pada urutan file, tapi tetap deterministik terhadap seed.
    rng.shuffle(order)
    order.sort(key=lambda kv: -len(kv[1]))

    assignment = np.empty(n, dtype=object)
    for _, idxs in order:
        counts = np.bincount(labels[idxs], minlength=len(CLASSES)).astype(float)
        # Split terpilih = yang defisitnya paling besar setelah klaster ini
        # masuk; dibobot per kelas supaya kelas minoritas ikut dijaga.
        best, best_score = None, None
        for s in SPLITS:
            deficit = quota[s] - current[s]
            score = float(np.sum(deficit * counts) / max(1.0, counts.sum()))
            if best_score is None or score > best_score:
                best, best_score = s, score
        current[best] += counts
        for i in idxs:
            assignment[i] = best
    return assignment


def random_assignment(
    labels: np.ndarray, val_frac: float, test_frac: float, seed: int
) -> np.ndarray:
    """Split acak per gambar, distratifikasi per kelas. Baseline pembanding."""
    assignment = np.empty(len(labels), dtype=object)
    for c in range(len(CLASSES)):
        idx = np.where(labels == c)[0].tolist()
        random.Random(seed + c).shuffle(idx)
        n = len(idx)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        for i in idx[:n_test]:
            assignment[i] = "test"
        for i in idx[n_test:n_test + n_val]:
            assignment[i] = "val"
        for i in idx[n_test + n_val:]:
            assignment[i] = "train"
    return assignment


def leakage_report(embeddings: np.ndarray, assignment: np.ndarray, threshold: float) -> dict:
    """Berapa persen gambar val/test yang punya wajah sama di train."""
    train_mask = assignment == "train"
    if not train_mask.any():
        return {}
    train_emb = embeddings[train_mask]
    out = {}
    for split in ("val", "test"):
        mask = assignment == split
        if not mask.any():
            continue
        best = (embeddings[mask] @ train_emb.T).max(axis=1)
        out[split] = round(float((best >= threshold).mean()) * 100, 2)
    return out


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT,
                    help=f"Folder hasil (default: {DEFAULT_OUT})")
    ap.add_argument("--zip", type=Path, default=None,
                    help="Pakai zip yang sudah diunduh manual, jangan unduh lagi.")
    ap.add_argument("--split-mode", choices=("identity", "random"), default="identity",
                    help="identity = satu orang tidak boleh muncul di dua split "
                         "(default, jujur). random = split acak per gambar "
                         "(baseline yang menggelembung, untuk pembanding).")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--sim-threshold", type=float, default=IDENTITY_SIM_THRESHOLD,
                    help="Cosine similarity minimum untuk menganggap dua wajah "
                         "orang yang sama saat mengklaster.")
    ap.add_argument("--margin", type=float, default=0.25,
                    help="Margin crop wajah. HARUS sama dengan FaceLandmarker.margin "
                         "dan FatiguePipeline agar training & inferensi sepadan.")
    ap.add_argument("--size", type=int, default=256,
                    help="Sisi terpanjang crop yang disimpan. Lebih besar dari "
                         "input model (224) supaya masih ada ruang RandomResizedCrop.")
    ap.add_argument("--no-crop", action="store_true",
                    help="Simpan gambar apa adanya tanpa deteksi wajah (debug).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out: Path = args.out
    zip_path = args.zip or (out.parent / "fatigue-dataset.zip")
    if args.zip and not args.zip.exists():
        raise SystemExit(f"[ERROR] File tidak ada: {args.zip}")
    if not args.zip:
        download_zip(zip_path)

    raw_dir = out.parent / "fatigue-raw"
    grouped = extract(zip_path, raw_dir)
    for label, files in grouped.items():
        print(f"[INFO] {label}: {len(files)} gambar mentah")

    # detect_width=None: deteksi selalu di resolusi penuh. Ini jalur offline
    # yang dijalankan sekali, jadi kecepatan tidak penting — sedangkan crop
    # yang identik antar-run itu penting, karena crop inilah yang jadi data
    # training. Runtime memakai default (640) untuk kecepatan; selisih beberapa
    # piksel di bbox sudah tercakup augmentasi RandomResizedCrop.
    detector = None if args.no_crop else FaceDetector(min_face=24, detect_width=None)
    # Embedder hanya diperlukan untuk mode identity; memuatnya 37 MB percuma
    # kalau penggunanya memang minta split acak.
    embedder = SFaceEmbedder() if args.split_mode == "identity" and detector else None
    if args.split_mode == "identity" and detector is None:
        raise SystemExit("[ERROR] --split-mode identity tidak bisa dipakai dengan --no-crop.")

    # ---- pass 1: crop + embed, tahan di memori (2.200 crop 256px ~ 400 MB) ----
    # Embedding di-cache: pass ini makan ~165 detik dan tidak berubah selama
    # margin/ukuran crop-nya sama, sedangkan --sim-threshold sering di-tuning
    # berulang kali. Tanpa cache, tiap percobaan ambang bayar ongkos penuh.
    cache_path = out.parent / f"fatigue-embcache-m{args.margin}-s{args.size}.npz"
    cache = None
    if embedder is not None and cache_path.exists():
        loaded = np.load(cache_path, allow_pickle=False)
        if len(loaded["labels"]) == sum(len(v) for v in grouped.values()):
            cache = loaded
            print(f"[INFO] Memakai cache embedding: {cache_path.name}")
        else:
            print("[INFO] Cache embedding basi (jumlah gambar berubah), hitung ulang.")

    print(f"[INFO] Memproses wajah (mode split: {args.split_mode})…")
    t0 = time.perf_counter()
    crops: list[np.ndarray] = []
    labels: list[int] = []
    embeddings: list[np.ndarray] = []
    stats = Counter()

    for label, files in grouped.items():
        class_idx = CLASSES.index(label)
        for src in files:
            img = cv2.imread(str(src))
            if img is None:
                stats["unreadable"] += 1
                continue

            crop, face = (None, None) if detector is None else crop_face(detector, img, args.margin)
            if crop is None:
                stats[f"nocrop_{label}"] += 1
                crop = img
            else:
                stats[f"cropped_{label}"] += 1

            if embedder is not None:
                if cache is not None:
                    vec = cache["embeddings"][len(embeddings)]
                else:
                    # Wajah tak terdeteksi tidak punya landmark untuk alignment;
                    # diberi vektor nol supaya ia jadi klaster sendiri (nol tidak
                    # pernah mencapai ambang kemiripan terhadap apa pun).
                    vec = (embedder.embed(img, face) if face is not None
                           else np.zeros(embedder.dim, dtype=np.float32))
                embeddings.append(vec)

            h, w = crop.shape[:2]
            scale = args.size / max(h, w)
            if scale < 1.0:
                crop = cv2.resize(
                    crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
                )
            crops.append(crop)
            labels.append(class_idx)

    labels_arr = np.asarray(labels, dtype=np.int64)
    print(f"[INFO] {len(crops)} gambar diproses dalam {time.perf_counter() - t0:.1f}s")

    # ---- pass 2: tentukan split ----
    emb_arr = np.stack(embeddings) if embeddings else None
    if emb_arr is not None and cache is None:
        np.savez_compressed(cache_path, embeddings=emb_arr, labels=labels_arr)
        print(f"[INFO] Cache embedding disimpan: {cache_path.name}")
    if args.split_mode == "identity":
        cluster_ids = cluster_identities(emb_arr, args.sim_threshold)
        n_clusters = int(cluster_ids.max()) + 1
        sizes = np.bincount(cluster_ids)
        print(f"[INFO] {n_clusters} klaster identitas "
              f"(terbesar {sizes.max()} gambar, {int((sizes == 1).sum())} klaster tunggal)")
        assignment = assign_clusters(
            cluster_ids, labels_arr, args.val_frac, args.test_frac, args.seed
        )
    else:
        cluster_ids = None
        assignment = random_assignment(labels_arr, args.val_frac, args.test_frac, args.seed)

    leak = leakage_report(emb_arr, assignment, 0.40) if emb_arr is not None else {}
    if leak:
        print("[INFO] Kebocoran identitas ke train (sim>=0.40): "
              + ", ".join(f"{k}={v}%" for k, v in leak.items()))

    # ---- pass 3: tulis ----
    if out.exists():
        shutil.rmtree(out)
    for split in SPLITS:
        for label in CLASSES:
            (out / split / label).mkdir(parents=True, exist_ok=True)

    counters = Counter()
    for i, (crop, class_idx) in enumerate(zip(crops, labels_arr.tolist())):
        label = CLASSES[class_idx]
        split = assignment[i]
        dst = out / split / label / f"{label}_{i:05d}.jpg"
        cv2.imwrite(str(dst), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        counters[f"{split}_{label}"] += 1

    manifest = {
        "source": f"kaggle:{KAGGLE_REF}",
        "license": "MIT",
        "classes": list(CLASSES),
        "split_mode": args.split_mode,
        "identity_sim_threshold": args.sim_threshold if args.split_mode == "identity" else None,
        "identity_clusters": int(cluster_ids.max()) + 1 if cluster_ids is not None else None,
        "face_cropped": not args.no_crop,
        "crop_margin": args.margin,
        "stored_max_side": args.size,
        "seed": args.seed,
        "splits": {
            split: {label: counters[f"{split}_{label}"] for label in CLASSES}
            for split in SPLITS
        },
        "crop_success": {label: stats[f"cropped_{label}"] for label in CLASSES},
        "crop_failed": {label: stats[f"nocrop_{label}"] for label in CLASSES},
        "unreadable": stats["unreadable"],
        "identity_leakage_pct": leak,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n[OK] Selesai dalam {time.perf_counter() - t0:.1f}s -> {out}")
    for split in SPLITS:
        row = manifest["splits"][split]
        print(f"     {split:5s} " + "  ".join(f"{k}={v}" for k, v in row.items()))
    print(f"     wajah terdeteksi: {sum(manifest['crop_success'].values())}, "
          f"dipakai utuh: {sum(manifest['crop_failed'].values())}")
    shutil.rmtree(raw_dir, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
