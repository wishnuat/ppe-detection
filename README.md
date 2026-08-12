# 🦺 PPE Detection — YOLOv8 + FastAPI + Streamlit

Sistem deteksi Alat Pelindung Diri (APD/PPE) berbasis **YOLOv8** dengan
pipeline lengkap: inference lokal (image / video / webcam), REST API via
**FastAPI**, dan demo UI via **Streamlit**. Siap dijalankan lewat Docker.

Dibangun sebagai **tugas akhir sertifikasi AI Super Class — track Computer
Vision**, melanjutkan proyek PPE Detection sebelumnya yang berbasis Roboflow
Serverless Inference API menjadi solusi yang **offline-capable** dan siap
diintegrasikan ke perangkat edge (CCTV / dashcam armada).

---

## 📌 Latar Belakang

Kepatuhan penggunaan APD di lingkungan kerja industri (tambang, konstruksi,
logistik, transportasi) adalah salah satu indikator utama Keselamatan &
Kesehatan Kerja (K3). Pengawasan manual tidak scalable, sehingga otomasi
berbasis Computer Vision menjadi solusi natural.

Proyek ini merupakan **evolusi** dari iterasi pertama saya yang memakai
Roboflow Serverless Inference (butuh koneksi internet & API call per frame),
menjadi arsitektur yang:

- **Inference lokal** (Ultralytics YOLOv8 `.pt`) — tidak bergantung API online.
- Punya **layer API** untuk integrasi ke sistem lain (dashboard armada,
  event logger, telegram bot alert, dll).
- Punya **UI demo** untuk validasi cepat & presentasi hasil ke stakeholder.
- **Container-ready** untuk deployment ke perangkat lapangan.

---

## 🎯 Class yang Dideteksi

Dataset: Roboflow `wishnus-workspace/ppe-detection-hyeuz-6cijw` **version 2** —
**17 kelas mentah**, yang dipetakan sistem menjadi **8 kategori kepatuhan**.

| Kategori kepatuhan | Label positif (model) | Label pelanggaran (model) |
|--------------------|-----------------------|----------------------------|
| helmet         | `head_helmet`     | `head_nohelmet`     |
| glasses        | `glasses`         | `No_Glasses`        |
| mask           | `face_mask`       | `face_nomask`       |
| glove          | `hand_glove`      | `hand_noglove`      |
| shoes          | `shoes`, `boots`  | `Barefoots`, `Sandals` |
| vest           | `vest`            | — (tidak ada di dataset) |
| ear_protection | `Ear-protection`  | `No_Ear-Protection` |
| harness        | `Harness`         | — (tidak ada di dataset) |

Kelas ke-17 adalah `person`, ikut terdeteksi tapi bukan APD sehingga tidak
masuk perhitungan compliance.

Pemetaan ini ada di `RAW_LABEL_MAP` pada `src/detector.py`. Setiap deteksi
mengembalikan `label` (mentah) **dan** `category` (kategori kepatuhan).
Status per kategori: ✅ **TERDETEKSI** / ⚠️ **PELANGGARAN** / ⚪ **TIDAK
TERDETEKSI**. Deteksi pelanggaran digambar dengan bounding box merah.

> Catatan: `vest` dan `harness` tidak punya label negatif di dataset ini,
> jadi keduanya tidak akan pernah berstatus PELANGGARAN — hanya
> TERDETEKSI / TIDAK TERDETEKSI.

### Metrics

Ada dua angka yang jangan sampai tertukar — keduanya melatih arsitektur dan
data yang berbeda:

| | Roboflow Train v2 | Training lokal (`models/best.pt`) |
|---|---|---|
| mAP@50 | **86.3%** | **65.5%** |
| mAP@50-95 | — | **38.9%** |
| Precision | **83.0%** | **67.6%** |
| Recall | **84.6%** | **62.8%** |
| Dievaluasi pada | — | split **test**, 948 gambar |
| Data train | 19.890 gambar | 2.500 gambar (subset) |
| Resolusi | — | 416 px |
| Epochs | — | 20 |
| Hardware | GPU (cloud Roboflow) | CPU 12th Gen i7-1265U, ~4 jam |

Angka Roboflow lebih tinggi karena dilatih di **dataset penuh** dengan GPU.
Model lokal sengaja dilatih pada subset 2.500 gambar supaya bisa selesai di
CPU dalam hitungan jam — tujuannya membuktikan pipeline-nya utuh dan
menghasilkan `.pt` yang bisa di-export ke OpenVINO, bukan mengejar SOTA.
**Untuk produksi, latih ulang di GPU dengan dataset penuh.**

Per-kelas (mAP@50 pada split validasi saat training), model lokal sudah kuat
di objek besar dan kontras
(`boots` 0.905 · `Barefoots` 0.905 · `person` 0.881 · `vest` 0.858) tapi
lemah di kelas negatif yang objeknya kecil dan ambigu
(`hand_noglove` 0.255 · `No_Ear-Protection` 0.310 · `No_Glasses` 0.411).
Pola ini masuk akal: "tangan tanpa sarung tangan" secara visual hanyalah
tangan biasa, jadi modelnya butuh jauh lebih banyak contoh daripada yang ada
di subset. Rincian lengkap ada di `runs/ppe/ppe-yolov8n-cpu/`.

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────┐    ┌────────────────────┐    ┌─────────────────┐
│  Input          │    │  Inference Engine  │    │  Output         │
│  ─────────      │    │  ────────────────  │    │  ─────────      │
│  • Image file   │───▶│  torch │ openvino  │───▶│  • Annotated    │
│  • Video file   │    │  int8  │ roboflow  │    │    image/video  │
│  • Webcam RTSP  │    │  ── build_detector │    │  • JSON         │
│  • HTTP upload  │    │  + compliance      │    │    (bbox, conf) │
│                 │    │    logic           │    │  • Compliance   │
└─────────────────┘    └─────────┬──────────┘    │    per class    │
                                 │               └────────┬────────┘
                       ┌─────────┴──────────┐             │
                       │  Delivery layer    │             │
                       │  ────────────────  │             │
                       │  FastAPI  (REST)   │◀────────────┘
                       │  Streamlit (UI)    │
                       │  CLI      (local)  │
                       └────────────────────┘
```

**Modul utama:**

```
PPE Detection/
├── src/
│   ├── detector.py       # PPEDetector (load YOLOv8, predict, render, filter kategori)
│   ├── openvino_detector.py # Backend OpenVINO IR (FP32/INT8), interface sama
│   ├── remote.py         # RoboflowDetector — backend serverless, interface sama
│   ├── backends.py       # build_detector() — pemilih backend untuk CLI/API/UI
│   └── cli.py            # CLI: python -m src.cli image|video|webcam
├── app/
│   ├── api.py            # FastAPI: /health, /predict, /predict/image
│   └── streamlit_app.py  # Streamlit demo UI (realtime + panel filter kategori)
├── scripts/
│   ├── download_dataset.py # Download dataset YOLOv8 dari Roboflow
│   ├── train.py            # Training lokal -> models/best.pt
│   ├── export_openvino.py  # .pt -> OpenVINO IR (FP32 + INT8 terkuantisasi)
│   ├── benchmark.py        # Ukur latency/FPS tiap backend -> docs/BENCHMARK.md
│   └── evaluate.py         # Ukur mAP tiap varian model  -> docs/METRICS.md
├── datasets/             # (dataset hasil export, tidak di-commit)
├── models/               # (weight .pt & IR disimpan di sini, tidak di-commit)
├── Dockerfile
├── docker-compose.yml    # Service: api (8000) + ui (8501)
├── requirements.txt
└── .env.example
```

---

## ⚙️ Setup & Install

### 1. Clone & buat virtualenv

```bash
git clone <repo-url>
cd "PPE Detection"

python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt
```

### 2. Konfigurasi environment

```bash
cp .env.example .env
# edit .env, isi ROBOFLOW_API_KEY
```

Variabel yang paling sering diubah:

| Variable | Default | Fungsi |
|----------|---------|--------|
| `INFERENCE_BACKEND` | `torch` | Backend default untuk FastAPI (`torch` / `openvino` / `openvino-int8` / `roboflow`) |
| `OPENVINO_DEVICE` | `CPU` | Device OpenVINO: `CPU`, `GPU` (iGPU Intel), `AUTO` |
| `OPENVINO_MODEL_DIR` | auto | Override lokasi folder IR |
| `MODEL_PATH` | `models/best.pt` | Weights untuk backend `torch` |
| `CONF_THRESHOLD` / `IOU_THRESHOLD` | `0.35` / `0.45` | Threshold deteksi & NMS |
| `ROBOFLOW_API_KEY` | — | Wajib untuk download dataset & backend `roboflow` |

> ⚠️ Jangan menamai variabel endpoint FastAPI sebagai `API_URL`. SDK
> `roboflow` membaca env var bernama `API_URL` sebagai base URL-nya sendiri,
> sehingga download dataset akan gagal connect. Project ini memakai
> `PPE_API_URL`.

### 3. Siapkan weights `models/best.pt`

> ⚠️ **Roboflow tidak menyediakan download file `.pt`** untuk model yang
> dilatih di platform mereka (Roboflow Train). Yang bisa diunduh adalah
> datasetnya. Jadi untuk inference offline, weights harus dilatih sendiri.

```bash
# 1) Ambil dataset (format YOLOv8) — ~22.7k gambar
python scripts/download_dataset.py

# 2) Latih. Dataset penuh praktis butuh GPU.
python scripts/train.py --epochs 50 --imgsz 640 --batch 16
# → runs/ppe/<name>/weights/best.pt, otomatis disalin ke models/best.pt
```

Smoke test cepat di CPU (bukan untuk produksi, hanya memvalidasi pipeline):

```bash
python scripts/train.py --epochs 3 --imgsz 320 --batch 8 --subset 300
```

**Perkiraan waktu training** (diukur di mesin CPU 12-core, tanpa CUDA):
~4 gambar/detik pada imgsz 320. Artinya dataset penuh ≈ 80 menit *per epoch*
di 320px, dan jauh lebih lama di 640px — training penuh di CPU tidak realistis.
Gunakan GPU (Google Colab / RunPod / mesin ber-CUDA) untuk run yang sebenarnya.

Kalau kamu sudah punya file `.pt` sendiri, cukup taruh di `models/best.pt`
(atau override lewat env `MODEL_PATH`) dan lewati langkah training.

### 4. Export ke OpenVINO (disarankan untuk deployment CPU/edge)

```bash
python scripts/export_openvino.py
# FP32 saja, tanpa perlu dataset kalibrasi:
python scripts/export_openvino.py --no-int8
```

Menghasilkan `models/best_openvino_model/` (FP32) dan
`models/best_int8_openvino_model/` (INT8). Lihat section berikutnya.

---

## ⚡ Backend Inference

Semua backend mewarisi interface `PPEDetector` yang sama, jadi CLI, FastAPI,
dan Streamlit tinggal memanggil `build_detector(...)` tanpa peduli mana yang
aktif. Pemilihnya ada di `src/backends.py`.

| Backend | Sumber model | Kapan dipakai |
|---------|--------------|---------------|
| `torch` (default) | `models/best.pt` | Referensi akurasi. Paling gampang — tidak butuh export. |
| `openvino` | `models/best_openvino_model/` | Deployment di CPU/iGPU Intel. Akurasi identik `.pt`, latency jauh lebih rendah. |
| `openvino-int8` | `models/best_int8_openvino_model/` | Edge box paling terbatas. Tercepat & paling kecil, dengan sedikit penurunan mAP. |
| `roboflow` | Serverless API | Baseline / saat weights lokal belum siap. Butuh internet. |

Alias lama tetap diterima: `local` dan `pytorch` → `torch`, `ov` → `openvino`,
`int8` → `openvino-int8`.

**Cara memilih:**

```bash
# CLI
python -m src.cli --backend openvino-int8 --device GPU image foto.jpg

# FastAPI (lewat environment)
INFERENCE_BACKEND=openvino uvicorn app.api:app --port 8000

# Streamlit → pilih di sidebar "Backend inference" + "Device OpenVINO"
```

`--device` hanya berlaku untuk backend OpenVINO: `CPU` (default), `GPU`
(iGPU Intel Iris Xe / UHD), atau `AUTO`.

### Kenapa INT8?

Quantization INT8 mengubah bobot & aktivasi dari float32 ke integer 8-bit.
Modelnya jadi ~4x lebih kecil dan jauh lebih cepat di CPU, tapi ada trade-off
akurasi. Prosesnya butuh **data kalibrasi** — NNCF menjalankan beberapa ratus
gambar dari split `val` untuk mengukur rentang aktivasi tiap layer — makanya
`export_openvino.py` menolak jalan kalau dataset belum di-download.

### Hasil benchmark

Diukur di **Intel Core i7-1265U** (12 core, tanpa GPU diskrit), model 416 px,
10 gambar berbeda × 50 iterasi. Latency = **end-to-end per frame**
(preprocess + inference + NMS + mapping ke objek `Detection`), bukan hanya
forward pass — karena itulah yang menentukan FPS sebenarnya di stream kamera.

| Backend | Device | Mean (ms) | p95 (ms) | FPS | Speedup |
|---------|--------|-----------|----------|-----|---------|
| `torch` | CPU | 155.8 | 233.7 | 6.4 | 1.00× |
| `openvino` | CPU | 84.4 | 135.1 | 11.8 | 1.85× |
| `openvino` | **GPU** | 22.9 | 33.3 | 43.7 | **6.81×** |
| `openvino-int8` | CPU | 70.0 | 149.3 | 14.3 | 2.22× |
| `openvino-int8` | **GPU** | 18.2 | 22.2 | **54.9** | **8.55×** |

Tiga hal yang menarik dari angka ini:

1. **iGPU-nya jauh lebih berpengaruh daripada quantization.** Pindah dari CPU
   ke iGPU Intel memberi ~3,7× — jauh lebih besar daripada lompatan FP32→INT8
   di CPU (~1,2×). Intel Iris Xe yang selama ini menganggur ternyata mesin
   inference yang layak.
2. **INT8 di CPU justru paling tidak stabil.** Mean-nya turun ke 70 ms, tapi
   p95-nya 149 ms — lebih buruk daripada FP32 CPU (135 ms). Untuk stream
   realtime, latency ekor seperti ini terasa sebagai frame drop yang tersendat.
   Di GPU sebaliknya: p95 22,2 ms, paling konsisten dari semuanya.
3. **Ukuran model turun 3,4×** — 11,7 MB (FP32) → 3,4 MB (INT8), relevan untuk
   edge box dengan storage terbatas.

### Berapa akurasi yang hilang?

Kecepatan tidak ada artinya kalau modelnya jadi bodoh. Diukur di **split test
yang belum pernah dilihat model** (948 gambar, 416 px):

| Varian | mAP@50 | mAP@50-95 | Precision | Recall | Δ mAP@50 |
|--------|--------|-----------|-----------|--------|----------|
| PyTorch FP32 | 0.6545 | 0.3886 | 0.6756 | 0.6282 | — |
| OpenVINO FP32 | 0.6547 | 0.3845 | 0.6682 | 0.6277 | **+0.02 pp** |
| OpenVINO INT8 | 0.6513 | 0.3820 | 0.6636 | 0.6298 | **−0.32 pp** |

- **OpenVINO FP32 secara efektif identik dengan PyTorch** (selisih +0.02 pp
  adalah noise pembulatan, bukan perbaikan nyata). Ini yang diharapkan — FP32
  hanya mengubah cara model dieksekusi, bukan bobotnya.
- **INT8 hanya membayar 0,32 poin mAP@50** untuk 8,55× kecepatan. Recall-nya
  bahkan sedikit lebih tinggi (0.6298 vs 0.6282); yang turun adalah precision,
  artinya INT8 sedikit lebih "berani" menghasilkan deteksi.

Untuk deteksi APD, trade-off ini menguntungkan: **melewatkan** pekerja tanpa
helm jauh lebih mahal daripada satu false positive yang bisa diverifikasi
supervisor.

**Rekomendasi:** `openvino-int8 @ GPU` untuk deployment (tercepat, p95 paling
konsisten, biaya akurasi 0,32 pp), `torch` saat development karena tidak butuh
export ulang tiap kali weights berubah.

Angka lengkap: [`docs/BENCHMARK.md`](docs/BENCHMARK.md) (`python scripts/benchmark.py`)
· [`docs/METRICS.md`](docs/METRICS.md) (`python scripts/evaluate.py`)

---

## 🚀 Menjalankan

### CLI (local)

```bash
# Satu gambar
python -m src.cli image path/to/foto.jpg

# Video file
python -m src.cli video path/to/video.mp4

# Webcam realtime (tekan 'q' berhenti, 's' snapshot)
python -m src.cli webcam --index 0 --save outputs/webcam_demo.mp4
```

Flag global:

| Flag | Fungsi |
|------|--------|
| `--backend torch` (default) | Inference offline pakai `models/best.pt` (alias: `local`) |
| `--backend openvino` / `openvino-int8` | Inference lewat OpenVINO IR — jauh lebih cepat di CPU/iGPU Intel |
| `--backend roboflow` | Inference via Roboflow Serverless (butuh internet) — berguna sebagai baseline atau saat weights lokal belum siap |
| `--device CPU\|GPU\|AUTO` | Hanya untuk backend OpenVINO. `GPU` = iGPU Intel |
| `--conf 0.4` | Confidence threshold |
| `--categories helmet,vest` | Hanya deteksi kategori tertentu; sisanya dibuang dari box, tabel, dan compliance |

Contoh — abaikan sarung tangan dan orang:

```bash
python -m src.cli --categories helmet,glasses,mask,shoes,vest,ear_protection,harness image foto.jpg
```

### FastAPI service

```bash
uvicorn app.api:app --reload --port 8000
```

Endpoints:

| Method | Path             | Deskripsi                                                    |
|--------|------------------|--------------------------------------------------------------|
| GET    | `/health`        | Health check + info model (backend aktif, device, path, jumlah class, list class) |
| POST   | `/predict`       | Multipart upload gambar → JSON `{detections, compliance, annotated_image_b64}` |
| POST   | `/predict/image` | Multipart upload gambar → langsung return PNG hasil deteksi |

Dokumentasi interaktif: **http://localhost:8000/docs**

Contoh request:

```bash
curl -X POST -F "file=@sample.jpg" http://localhost:8000/predict | jq .compliance
```

Contoh response `/predict` (dipotong):

```json
{
  "width": 640,
  "height": 480,
  "compliance": {
    "helmet": "PELANGGARAN",
    "vest": "TERDETEKSI",
    "mask": "TIDAK TERDETEKSI"
  },
  "detections": [
    {
      "label": "head_nohelmet",
      "category": "helmet",
      "confidence": 0.7836,
      "bbox": [223, 100, 402, 189],
      "is_violation": true
    }
  ],
  "annotated_image_b64": "iVBORw0KGgo..."
}
```

`label` = kelas mentah model, `category` = kategori kepatuhan hasil pemetaan.

### Streamlit demo UI

```bash
streamlit run app/streamlit_app.py
```

Buka **http://localhost:8501**. Sidebar berisi:

- **Mode input** — Gambar / Video / Webcam
- **Backend inference** — OpenVINO INT8 / OpenVINO FP32 / PyTorch / Roboflow
  serverless. Detector di-cache per (backend, device) karena compile OpenVINO
  makan beberapa detik dan Streamlit me-rerun script tiap interaksi.
- **Device OpenVINO** — CPU / GPU (iGPU Intel) / AUTO, muncul saat backend
  OpenVINO dipilih
- **Confidence threshold**
- **🎚️ Kategori yang dideteksi** — checkbox per kategori. Uncheck *Sarung
  tangan*, misalnya, dan deteksi glove hilang dari bounding box, tabel
  detail, dan kartu compliance sekaligus. Ada tombol *Pilih semua* /
  *Kosongkan*.

Mode **Webcam** punya dua sub-mode:

| Sub-mode | Cara kerja | Kapan dipakai |
|----------|-----------|----------------|
| **Realtime** | Server Streamlit membaca kamera langsung dan men-stream frame teranotasi ke browser, lengkap dengan FPS, banner pelanggaran, dan kartu compliance yang ter-update tiap frame. Ada kontrol *Camera index*, *Batas FPS*, tombol Mulai/Stop. | Streamlit jalan di mesin yang sama dengan kamera (laptop, edge box) |
| **Snapshot** | Memakai kamera **browser** via `st.camera_input` | Streamlit di-deploy ke server remote yang tidak punya kamera |

> Filter kategori diterapkan di layer detector (`enabled_categories`), bukan
> hanya di UI — jadi CLI, REST API, dan UI berperilaku sama.

### Docker (semua sekaligus)

```bash
docker compose up --build
```

- API   → http://localhost:8000
- UI    → http://localhost:8501

`models/` dan `outputs/` di-mount sebagai volume, jadi weight tidak perlu
di-bake ke image dan hasil deteksi tetap persistent di host. Folder IR
OpenVINO ikut ter-mount lewat volume yang sama, jadi `INFERENCE_BACKEND=openvino`
di `.env` langsung berlaku di dalam container — tanpa rebuild image.

> Backend `openvino` di container hanya jalan di **CPU**. Akses iGPU Intel dari
> dalam container butuh passthrough `/dev/dri` + driver compute di image, yang
> belum disiapkan di sini. Untuk uji `--device GPU`, jalankan langsung di host.

---

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
pytest tests -q
```

`tests/test_detector.py` menguji taksonomi label dan logika compliance
(termasuk aturan "satu pelanggaran mengalahkan deteksi positif") tanpa perlu
file `.pt`, jadi test tetap jalan di CI yang tidak punya weights.

`tests/test_backends.py` menguji dua hal, juga tanpa perlu weights maupun
runtime OpenVINO:

1. **Pemilihan backend** — normalisasi nama & alias lama, penolakan nama tidak
   dikenal, pembacaan env `INFERENCE_BACKEND`, dan bahwa error "model belum
   di-export" menyebut cara memperbaikinya.
2. **Pre/post-processing OpenVINO** — `letterbox` (ukuran output, aspect ratio
   terjaga, roundtrip koordinat, warna padding) dan `nms` (supresi box
   bertumpuk, box terpisah, urutan skor, input kosong, efek threshold). Ini
   bagian yang ditulis manual karena OpenVINO tidak membawa post-processing
   YOLO seperti Ultralytics, jadi paling rawan salah.

---

## 🩺 Troubleshooting

| Gejala | Sebab & solusi |
|--------|----------------|
| `download_dataset.py` gagal connect ke `localhost:8000` | Ada env var bernama `API_URL`. SDK roboflow memakainya sebagai base URL. Rename jadi `PPE_API_URL`. |
| `ModuleNotFoundError: No module named 'src'` saat `streamlit run` | Streamlit menaruh `app/` di `sys.path`, bukan root project. Sudah ditangani oleh bootstrap `PROJECT_ROOT` di `app/streamlit_app.py`. |
| `FileNotFoundError: Model tidak ditemukan di models/best.pt` | Weights belum ada. Jalankan `scripts/download_dataset.py` lalu `scripts/train.py`, atau taruh `.pt` sendiri di `models/`. |
| `401 Unauthorized api_key` dari Roboflow | `.env` tidak terbaca (python-dotenv mencari relatif ke lokasi file pemanggil). Jalankan script dari root project atau pakai `load_dotenv(PROJECT_ROOT / ".env")`. |
| `docker compose` error `dockerDesktopLinuxEngine ... cannot find the file` | Docker Desktop terpasang tapi engine belum jalan. Start Docker Desktop dulu. |
| Training di CPU sangat lambat | Normal. Pakai `--subset` untuk smoke test, dan GPU untuk training sungguhan. |
| Training terputus di tengah jalan | Jangan mulai dari nol. `python scripts/train.py --resume --name <nama-run>` melanjutkan dari `last.pt`, lengkap dengan state optimizer. |
| `FileNotFoundError` menyebut `export_openvino` | Backend OpenVINO dipilih tapi IR-nya belum di-export. Jalankan `python scripts/export_openvino.py`. |
| Export INT8 gagal: "butuh dataset kalibrasi" | Quantization perlu gambar nyata untuk mengukur rentang aktivasi. Download dataset dulu, atau pakai `--no-int8`. |
| `--device GPU` gagal / tidak terdeteksi | iGPU Intel butuh driver yang dikenali OpenVINO. Fallback ke `CPU` atau `AUTO`. |
| `ignoring corrupt image/label: labels mix segment and detection rows` | Sebagian anotasi dataset ini berformat segmentasi (polygon), bukan bounding box. Ultralytics melewatinya; ~1% gambar hilang dan training tetap jalan. |

---

## 📸 Screenshot / Demo

Contoh output renderer — tiga pelanggaran terdeteksi sekaligus
(`head_nohelmet`, `No_Glasses`, `face_nomask`), semuanya digambar merah:

![Demo deteksi pelanggaran APD](docs/screenshots/demo_detection.jpg)

Compliance status untuk frame di atas:

```json
{ "helmet": "PELANGGARAN", "glasses": "PELANGGARAN", "mask": "PELANGGARAN",
  "glove": "TIDAK TERDETEKSI", "shoes": "TIDAK TERDETEKSI",
  "vest": "TIDAK TERDETEKSI", "ear_protection": "TIDAK TERDETEKSI",
  "harness": "TIDAK TERDETEKSI" }
```

_Screenshot berikut menyusul — simpan di `docs/screenshots/` lalu tampilkan di sini._

| Mode | Screenshot |
|------|------------|
| Streamlit — Image detection | `docs/screenshots/ui_image.png` |
| Streamlit — Compliance card | `docs/screenshots/ui_compliance.png` |
| FastAPI Swagger UI | `docs/screenshots/api_swagger.png` |
| Webcam realtime | `docs/screenshots/webcam.gif` |

---

## 🗺️ Rencana Pengembangan Lanjutan

Roadmap ini terkait langsung dengan pekerjaan saya di bidang **fleet GPS &
dashcam**:

1. **Integrasi ke stream dashcam armada** — konsumsi RTSP/HLS dari kamera
   in-cabin & side-camera, deteksi kepatuhan APD driver/kernet secara realtime.
2. **Integrasi ke CCTV loading dock / warehouse** — batching di edge box
   (Jetson Nano / Orin) supaya bandwidth ke cloud minimal.
3. **Event & alert pipeline** — pelanggaran (`head_nohelmet`, `hand_noglove`, dst)
   dikirim ke dashboard fleet management + notifikasi Telegram/WA ke supervisor
   HSE.
4. **Historical reporting** — agregasi compliance rate per unit / per shift /
   per site untuk audit K3.
5. **Retraining berkala** — feedback loop: false positive/negative dari
   lapangan → labelling di Roboflow → auto-retrain via Roboflow Train / RunPod.
6. **On-device optimization** — ✅ **OpenVINO FP32 + INT8 sudah jalan** (lihat
   `docs/BENCHMARK.md`), yang langsung relevan untuk edge box berbasis Intel.
   Berikutnya: **ONNX / TensorRT / NCNN** untuk Jetson dan dashcam Android.

---

## 🙏 Credits

- Dataset & training platform: [Roboflow](https://roboflow.com)
- Model architecture: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Serving: [FastAPI](https://fastapi.tiangolo.com), [Streamlit](https://streamlit.io)
- Dibuat sebagai tugas akhir **AI Super Class — Computer Vision Track**
