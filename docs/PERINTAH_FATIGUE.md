# Perintah siap-tempel — Fatigue & Absensi

Semua perintah di bawah ditulis untuk **PowerShell / Windows**, dan diawali
dengan dua baris yang sama:

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate
```

Setelah venv aktif (prompt berubah jadi `(venv)`), langsung tempel salah satu
blok di bawah. Penjelasan lengkap tiap komponen ada di [`FATIGUE.md`](FATIGUE.md).

---

## 0. Sekali saja — pastikan siap

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate

python -m src.fatigue.assets --verify
```

Kalau ada baris `[MISS]` atau `[BAD]`, unduh ulang:

```powershell
python -m src.fatigue.assets
```

Kalau `import mediapipe` gagal, pasang dengan `--no-deps` — **wajib**, karena
dependency-nya akan menimpa `opencv-python 4.10` dan merusak modul APD:

```powershell
pip install --no-deps mediapipe==1.0.1
pip install "absl-py>=2.0" "sounddevice~=0.5"
python -c "import cv2, mediapipe; print(cv2.__version__)"   # harus 4.10.0
```

---

## 1. UI Streamlit — cara paling gampang mencoba

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate
streamlit run app/streamlit_app.py
```

Di sidebar, pilih **Fatigue & absensi**. Lalu:

- **Karyawan** → daftarkan wajah Anda (unggah 5–10 foto dari sudut berbeda)
- **Monitor** → pilih Webcam → **Mulai**
- **Log absensi** → lihat catatan kehadiran + unduh CSV

> Beberapa detik pertama level akan `TIDAK_DIKETAHUI` — itu proses kalibrasi
> mata, bukan kegagalan. Untuk melihat level berubah jadi `LELAH`/`KRITIS`,
> coba pejamkan mata 3–4 detik di depan kamera.

---

## 2. CLI webcam — realtime tanpa browser

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate
python -m src.fatigue.cli webcam
```

Tekan `q` untuk berhenti, `s` untuk snapshot.

Variasi yang berguna:

```powershell
# kamera lain + rekam sesinya
python -m src.fatigue.cli webcam --index 1 --save outputs/uji_fatigue.mp4

# tanpa absensi — tidak memuat model wajah 37 MB, lebih cepat start
python -m src.fatigue.cli --no-attendance webcam

# CNN mati secara DEFAULT (terukur tidak andal di wajah kamera nyata).
# Nyalakan hanya setelah model dilatih ulang dengan frame kamera Anda:
python -m src.fatigue.cli --classifier webcam

# pakai OpenVINO (2-3x lebih cepat, akurasi identik)
python -m src.fatigue.cli --classifier-backend openvino webcam
```

---

## 3. Daftarkan wajah

Lewat webcam — tekan `SPASI` tiap kali mau mengambil foto:

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate
python scripts/enroll_faces.py --webcam --id EMP001 --name "Budi Santoso" --department Produksi
```

Lewat folder — **`data\faces\`**, satu subfolder per karyawan. Nama foldernya
menjadi `employee_id`:

```
data\faces\
├─ EMP001\
│  ├─ name.txt          (opsional — isi: "Budi Santoso")
│  ├─ depan.jpg
│  ├─ serong-kiri.jpg
│  └─ serong-kanan.jpg
└─ EMP002\
   ├─ name.txt
   └─ foto1.jpg
```

```powershell
mkdir data\faces\EMP001
# salin foto-fotonya ke situ, lalu:
python scripts/enroll_faces.py --dir data/faces --department Produksi
```

Nama karyawan diambil dari `name.txt` di dalam foldernya; kalau tidak ada,
dipakai nama foldernya sendiri (jadi `EMP002` di contoh atas akan tercatat
bernama "EMP002"). Format gambar yang diterima: `.jpg .jpeg .png .bmp .webp` —
file lain, termasuk `name.txt`, diabaikan.

Foto **ditolak** kalau: wajah tidak terdeteksi, ada lebih dari satu wajah,
wajah lebih kecil dari 80 px, atau nyaris identik dengan foto yang sudah
diterima. Foto yang agak buram hanya **diperingatkan**, tetap dipakai. Aturan
yang sama persis berlaku di UI Streamlit dan endpoint API.

> `data\` ada di `.gitignore` dan `.dockerignore` — foto dan database biometrik
> tidak akan pernah ikut ter-commit atau ter-bake ke dalam image.

Lihat & hapus:

```powershell
python scripts/enroll_faces.py --list
python scripts/enroll_faces.py --delete EMP001
```

---

## 4. Analisis file video / gambar

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate

# video — hasilnya di outputs/, plus ringkasan level terburuk per orang
python -m src.fatigue.cli video "C:\path\ke\rekaman.mp4" --stride 2

# satu gambar — hanya cek deteksi wajah & identitas
# (level kelelahan mustahil dari satu frame, dan CLI-nya bilang begitu)
python -m src.fatigue.cli image frame.jpg
```

---

## 5. API FastAPI

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate
uvicorn app.api:app --reload --port 8000
```

Buka <http://localhost:8000/docs> — Swagger, endpoint fatigue ada di grup
`fatigue`. Uji cepat dari PowerShell lain:

```powershell
curl.exe http://localhost:8000/fatigue/health
curl.exe -F "file=@frame.jpg" "http://localhost:8000/fatigue/analyze?session_id=cam1"
curl.exe http://localhost:8000/fatigue/employees
curl.exe http://localhost:8000/fatigue/attendance
```

> Kirim frame berurutan dengan `session_id` yang sama supaya riwayat temporalnya
> terkumpul. Satu request lepas akan selalu menjawab `TIDAK_DIKETAHUI`.

---

## 6. Test

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate

pytest tests -q                    # semua (180 test)
pytest tests -k fatigue -q         # modul fatigue saja (129 test)
pytest tests/test_fatigue_temporal.py -v
```

---

## 7. Melatih ulang dari nol

Hanya perlu kalau Anda mau melatih dengan data sendiri atau mereproduksi
angkanya. Urutannya penting.

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate

# unduh dataset Kaggle + crop wajah + split per identitas (~3 menit)
python scripts/prepare_fatigue_dataset.py

# latih (~66 menit di CPU i7 12-thread, tanpa GPU)
python scripts/train_fatigue.py --arch mobilenet_v3_large --epochs 30

# export ke ONNX + OpenVINO FP32 + INT8
python scripts/export_fatigue.py

# pilih titik operasi (tanpa training ulang) — lihat dulu, baru terapkan
python scripts/tune_fatigue_threshold.py
python scripts/tune_fatigue_threshold.py --criterion recall-floor --recall-floor 0.93 --apply

# metrik di test set, ketiga backend berdampingan
python scripts/evaluate_fatigue.py --batch-size 1
```

Bandingkan baseline yang menggelembung (split acak, ada kebocoran identitas)
dengan yang jujur:

```powershell
python scripts/prepare_fatigue_dataset.py --split-mode random
python scripts/train_fatigue.py --epochs 30
# lalu kembalikan: python scripts/prepare_fatigue_dataset.py
```

---

## 8. Benchmark kecepatan di mesin Anda

```powershell
cd "C:\Users\HP\Documents\PPE Detection"
venv\Scripts\activate

python scripts/benchmark_fatigue.py --source frame.jpg
python scripts/benchmark_fatigue.py --source frame.jpg --faces 4
python scripts/benchmark_fatigue.py --source frame.jpg --backend openvino-int8
```

Hasilnya ke `outputs/fatigue/benchmark*.json`, terurai per komponen.

---

## Kalau ada yang tidak beres

| Gejala | Penyebab & solusi |
|---|---|
| `ModuleNotFoundError: mediapipe` | Belum dipasang. Lihat bagian 0 — **wajib** `--no-deps`. |
| `cv2.__version__` jadi `5.x` | mediapipe dipasang tanpa `--no-deps` dan menimpa opencv. Perbaiki: `pip uninstall -y opencv-contrib-python opencv-python-headless; pip install opencv-python==4.10.0.84` |
| `Checkpoint fatigue tidak ditemukan` | Belum dilatih. Jalankan bagian 7, atau pakai `--no-classifier` untuk mode sinyal perilaku saja. |
| Level selalu `TIDAK_DIKETAHUI` | Wajah belum terlihat ≥ 5 detik, atau < 40% frame menghasilkan landmark. Dekatkan wajah ke kamera dan hadap depan. |
| Webcam gagal dibuka | Coba `--index 1` atau `--index 2`. Pastikan aplikasi lain (Zoom/Teams) tidak sedang memakai kamera. |
| Streamlit lambat / FPS rendah | Wajar — lihat [Performa](FATIGUE.md#performa). Turunkan "Batas FPS" di sidebar. |
| Level naik padahal saya segar | Pastikan toggle "Pakai CNN penampakan wajah" **mati** (default). CNN terukur menandai 59% wajah nyata sebagai lelah — lihat [alasannya](FATIGUE.md#cnn-dimatikan-secara-default--inilah-alasannya). |
