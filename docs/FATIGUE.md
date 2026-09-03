# Deteksi Fatigue Pekerja & Absensi Wajah

Modul tambahan di atas sistem PPE Detection yang sudah ada. Dari aliran CCTV
yang sama, modul ini menjawab dua pertanyaan berbeda:

1. **Siapa yang ada di sini?** — absensi berbasis pengenalan wajah
2. **Bagaimana kondisinya?** — deteksi kelelahan dari penampakan wajah + perilaku

Berjalan sepenuhnya offline di CPU. Tidak ada layanan cloud, tidak ada API key.

---

## Daftar isi

- [Kenapa dibangun seperti ini](#kenapa-dibangun-seperti-ini)
- [Arsitektur](#arsitektur)
- [Instalasi](#instalasi)
- [Alur kerja lengkap](#alur-kerja-lengkap)
- [Dataset & metodologi](#dataset--metodologi)
- [Model & hasil](#model--hasil)
- [Cara sistem memutuskan level kelelahan](#cara-sistem-memutuskan-level-kelelahan)
- [Absensi](#absensi)
- [Pemakaian](#pemakaian)
- [Konfigurasi](#konfigurasi)
- [Performa](#performa)
- [Batasan yang perlu diketahui](#batasan-yang-perlu-diketahui)
- [Privasi & kepatuhan](#privasi--kepatuhan)
- [Struktur file](#struktur-file)

---

## Kenapa dibangun seperti ini

Cara paling gampang membangun "fatigue detection" adalah melatih CNN pada
dataset wajah lelah/tidak lelah, lalu menampilkan keluarannya per frame.
Pendekatan itu menghasilkan angka akurasi yang bagus di laporan dan sistem yang
tidak bisa dipakai di lapangan. Ada tiga alasannya, dan tiga alasan itulah yang
membentuk seluruh desain modul ini.

**Pertama: kelelahan bukan properti satu frame.**
Orang yang sedang berkedip dan orang yang sedang tertidur terlihat identik
dalam satu gambar diam. Yang membedakan keduanya adalah *berapa lama* matanya
tertutup. Karena itu keluaran CNN tidak pernah dipakai langsung — ia masuk ke
jendela geser bersama PERCLOS, laju kedip, menguap, dan gerakan kepala
([`temporal.py`](../src/fatigue/temporal.py)).

**Kedua: satu ambang tidak cocok untuk semua orang.**
EAR (eye aspect ratio) seseorang bermata sipit saat terjaga penuh bisa lebih
rendah daripada EAR orang bermata besar saat setengah mengantuk. Ambang absolut
tunggal dijamin salah untuk sebagian karyawan. Sistem ini mengkalibrasi ambang
mata per orang dari beberapa detik pertama pengamatan.

**Ketiga: dataset publik wajah "lelah" penuh kebocoran identitas.**
Dataset Kaggle yang dipakai dikumpulkan dari internet, jadi satu orang muncul
di banyak gambar. Pada split acak biasa, **32% gambar test punya foto orang
yang sama di train**. Model bisa mendapat akurasi tinggi hanya dengan menghafal
wajah — "orang ini biasanya berlabel lelah" — dan angkanya tidak berarti apa
pun untuk karyawan yang belum pernah dilihat. Modul ini memecah data per
identitas wajah, bukan per gambar.

Akibat dari ketiganya: angka akurasi di dokumen ini lebih rendah daripada yang
biasa dilaporkan untuk dataset yang sama. Itu memang tujuannya.

---

## Arsitektur

```
                       ┌──────────────┐
        frame CCTV ───►│  FaceDetector│  YuNet (227 KB, cv2.FaceDetectorYN)
                       └──────┬───────┘
                              │  bbox + 5 landmark
              ┌───────────────┼───────────────────┐
              ▼               ▼                   ▼
      ┌───────────────┐ ┌──────────────┐ ┌─────────────────┐
      │ FaceEmbedder  │ │FaceLandmarker│ │FatigueClassifier│
      │ SFace 128-d   │ │ 478 titik +  │ │ MobileNetV3     │
      │ / ArcFace     │ │ 52 blendshape│ │ (crop wajah)    │
      └───────┬───────┘ └──────┬───────┘ └────────┬────────┘
              │                │                  │
              ▼                ▼                  │
      ┌───────────────┐ ┌──────────────┐          │
      │AttendanceBook │ │FatigueSignals│          │
      │ SQLite        │ │ EAR/MAR/pose │          │
      │ siapa ini?    │ └──────┬───────┘          │
      └───────┬───────┘        │                  │
              │                ▼                  │
              │      ┌───────────────────┐        │
              └─────►│  PersonTracker    │◄───────┘
             identitas│ jendela 60 dtk:  │  skor per-frame
                      │ PERCLOS, kedip,  │
                      │ menguap, angguk  │
                      └─────────┬─────────┘
                                ▼
                      ┌───────────────────┐
                      │  FatigueFusion    │  skor tertimbang + aturan keras
                      │  + histeresis     │  + penjelasan
                      └─────────┬─────────┘
                                ▼
                  SEGAR / WASPADA / LELAH / KRITIS
```

Empat model, semuanya CPU dan offline:

| Komponen | Model | Ukuran | Sumber |
|---|---|---|---|
| Deteksi wajah | YuNet 2023mar | 227 KB | OpenCV Zoo (MIT) |
| Pengenalan wajah | SFace 2021dec | 37 MB | OpenCV Zoo (Apache-2.0) |
| Landmark & blendshape | MediaPipe FaceLandmarker | 3,8 MB | Google (Apache-2.0) |
| Classifier fatigue | MobileNetV3-Large | 17 MB (INT8: 4,9 MB) | dilatih di sini (dataset MIT) |

YuNet dan SFace jalan lewat `cv2.FaceDetectorYN` / `cv2.FaceRecognizerSF` yang
sudah ada di `opencv-python` — **nol dependency baru** untuk deteksi dan
pengenalan wajah. Bobotnya diunduh otomatis dengan verifikasi sha256
([`assets.py`](../src/fatigue/assets.py)).

---

## Instalasi

```bash
# 1) dependency (dari root project)
pip install -r requirements.txt

# PENTING: mediapipe harus dipasang --no-deps.
# Dependency-nya menarik opencv-contrib-python 5.x yang MENIMPA
# opencv-python 4.10 dan mengubah API yang dipakai src/detector.py.
pip install --no-deps mediapipe==1.0.1
pip install "absl-py>=2.0" "sounddevice~=0.5"

# 2) unduh bobot pihak ketiga (41 MB, sekali saja, terverifikasi sha256)
python -m src.fatigue.assets

# 3) cek
python -m src.fatigue.assets --verify
```

Backend face recognition alternatif (opsional, lebih akurat pada pose ekstrem):

```bash
pip install insightface onnxruntime
# lalu set EMBEDDER_BACKEND=insightface
```

---

## Alur kerja lengkap

```bash
# --- melatih classifier fatigue ---
python scripts/prepare_fatigue_dataset.py     # unduh + crop + split per identitas
python scripts/train_fatigue.py               # 66 menit di CPU i7-1265U, tanpa GPU
python scripts/evaluate_fatigue.py            # metrik di test set, per backend
python scripts/export_fatigue.py              # ONNX + OpenVINO FP32 + INT8
python scripts/tune_fatigue_threshold.py      # pilih titik operasi (tanpa training ulang)

# --- mendaftarkan karyawan ---
python scripts/enroll_faces.py --dir data/faces
python scripts/enroll_faces.py --webcam --id EMP001 --name "Budi Santoso"
python scripts/enroll_faces.py --list

# --- menjalankan ---
python -m src.fatigue.cli webcam              # realtime
python -m src.fatigue.cli video rekaman.mp4   # analisis rekaman
streamlit run app/streamlit_app.py            # UI (pilih modul "Fatigue & absensi")
uvicorn app.api:app --reload                  # API di /fatigue/*
```

---

## Dataset & metodologi

**Sumber:** [`rihabkaci99/fatigue-dataset`](https://www.kaggle.com/datasets/rihabkaci99/fatigue-dataset)
— 2.200 gambar wajah, MIT, seimbang 1.100 `Fatigue` / 1.100 `NonFatigue`,
dikumpulkan dari internet. Bisa diunduh anonim, tidak butuh API key.

Dataset mentahnya tidak dipakai apa adanya. Dua transformasi penting:

### 1. Gambar dipotong ke wajah

Saat runtime, classifier hanya pernah melihat crop wajah keluaran YuNet dengan
margin 25%. Melatihnya pada foto utuh berisi bahu, latar, dan kadang dua orang
menghasilkan distribusi training yang berbeda dari distribusi inferensi:
akurasi validasi bagus, akurasi lapangan jatuh.

Crop memakai detektor, margin, dan interpolasi yang sama persis dengan runtime.
Definisi preprocessing-nya ada di satu tempat
([`classifier.preprocess_bgr`](../src/fatigue/classifier.py)) dan dipakai
bersama oleh training dan inferensi — bukan ditulis dua kali.

Hasil: **2.199 dari 2.200** gambar berhasil terdeteksi wajahnya. Satu yang
gagal dipakai utuh dan dicatat di manifest.

### 2. Split dikelompokkan per identitas, bukan per gambar

Ini yang paling menentukan kejujuran angkanya.

Semua wajah di-embed dengan SFace, lalu diklaster dengan union-find
single-linkage pada cosine similarity. Satu klaster utuh masuk ke satu split
saja. Ambang klaster (0.40) **sengaja disamakan** dengan ambang "orang yang
sama" milik absensi, sehingga nol kebocoran terjamin secara konstruksi: kalau
ada pasangan lintas-split dengan similarity ≥ 0.40, penutup transitif pasti
sudah menyatukannya lebih dulu.

Ambangnya dipilih setelah diukur, bukan ditebak:

| Ambang klaster | Klaster | Klaster terbesar | Bocor ke val | Bocor ke test |
|---:|---:|---:|---:|---:|
| 0.36 | 780 | **1.214** (55% data) | 0,0% | 0,0% |
| **0.40** | **1.522** | **116** (5% data) | **0,0%** | **0,0%** |
| 0.42 | 1.706 | 24 | 10,9% | 10,0% |
| 0.45 | 1.806 | 9 | 16,4% | 13,6% |
| 0.50 | 1.889 | 7 | 17,3% | 19,1% |

Di bawah 0.40, penutup transitif mulai menggumpal — banyak orang berbeda
terangkai jadi satu klaster raksasa yang memaksa seperempat dataset masuk ke
satu split. Di atas 0.40, kebocoran muncul. 0.40 adalah satu-satunya titik yang
memberi keduanya.

**Baseline pembanding tetap tersedia:** `--split-mode random` mereproduksi
split acak yang menggelembung itu, kalau ingin melihat sendiri berapa besar
selisihnya.

### Hasil split

| Split | nonfatigue | fatigue | Total |
|---|---:|---:|---:|
| train | 770 | 770 | 1.540 |
| val | 165 | 165 | 330 |
| test | 165 | 165 | 330 |

Seimbang sempurna, 1.522 klaster identitas, **kebocoran identitas 0,0%** ke val
maupun test.

---

## Model & hasil

### Konfigurasi training

| Aspek | Pilihan | Alasan |
|---|---|---|
| Backbone | MobileNetV3-Large (ImageNet) | Rasio akurasi/kecepatan terbaik untuk CPU realtime |
| Input | 224×224 | Standar backbone; crop disimpan 256px agar ada ruang RandomResizedCrop |
| Optimizer | AdamW, LR diskriminatif | Kepala baru butuh LR 10× backbone yang sudah bagus |
| Schedule | Cosine annealing | — |
| Loss | CrossEntropy, label smoothing 0.05 | Label dataset internet tidak sempurna; smoothing mencegah model terlalu yakin |
| EMA bobot | decay 0.999 | Pada 1.540 gambar, bobot per-epoch berayun keras |
| Seleksi model | ROC-AUC validasi | Akurasi bergantung ambang yang belum ditentukan saat itu; AUC tidak |
| Ambang keputusan | Youden's J di val | Bukan 0.5 |

**Augmentasi:** RandomResizedCrop(0.65–1.0), horizontal flip, ColorJitter,
RandomGrayscale(0.15), rotasi ±12°, RandomErasing(0.25).

Grayscale dan jitter warna agresif dipakai karena CCTV pabrik pada malam hari
nyaris monokrom — model tidak boleh bergantung pada warna kulit atau suhu warna
lampu. RandomErasing meniru oklusi nyata: masker, tangan mengusap mata, helm
yang turun.

Vertical flip dan rotasi besar **tidak** dipakai: wajah terbalik bukan
distribusi yang akan pernah ditemui kamera terpasang, dan melatihnya hanya
memboroskan kapasitas model.

### Ambang keputusan di-tune, bukan diasumsikan

Ambang dipilih di **validation set**; test set hanya dilaporkan sesudahnya.

Kriteria awalnya Youden's J (`TPR − FPR`), dan itu keliru. Youden mengasumsikan
biaya false negative dan false positive setara — sedangkan untuk sistem
peringatan kelelahan, melewatkan pekerja yang mengantuk jelas lebih mahal
daripada satu alarm palsu. Asumsi simetris itu bertentangan dengan alasan
sistem ini dibangun. Diukur: Youden memberi 19 false negative di test set,
sementara ambang yang lebih rendah memberi 9 dengan akurasi yang hampir sama.

Kriteria default sekarang `recall-floor` — presisi tertinggi di antara ambang
yang recall validasinya masih ≥ 0,93. Ambang terpilih: **0,2858**.

`scripts/tune_fatigue_threshold.py` memilih ulang titik ini tanpa melatih
ulang, dan menampilkan keempat kriteria berdampingan supaya pilihannya terlihat
alih-alih diyakini:

| Kriteria | Ambang | VAL recall | VAL FN | TEST recall | TEST akurasi | TEST FN | TEST FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| cost (FN 3× FP) | 0,2184 | 0,9576 | 7 | 0,9455 | 0,8818 | 9 | 30 |
| **recall-floor 0,93** | **0,2858** | **0,9455** | **9** | **0,9455** | **0,8848** | **9** | **29** |
| youden | 0,5065 | 0,8848 | 19 | 0,9152 | 0,9152 | 14 | 14 |
| f1 | 0,5065 | 0,8848 | 19 | 0,9152 | 0,9152 | 14 | 14 |

Akurasi turun (0,9152 → 0,8848) dan itu memang harganya: 15 alarm palsu
tambahan ditukar dengan 5 pekerja mengantuk yang tidak lagi terlewat. Pada
sistem keselamatan, itu arah pertukaran yang benar — dan pada sistem ini
khususnya, karena alarm palsu per-frame masih harus melewati agregasi 60 detik
plus histeresis sebelum sampai ke operator, sedangkan recall yang hilang tidak
bisa dipulihkan di mana pun.

> **Ambang ini tidak mengendalikan perilaku alarm.** `FatiguePipeline` memberi
> fusi probabilitas mentah classifier sebagai bukti kontinu, bukan keputusan
> biner — membinerkannya lebih dulu hanya membuang informasi. Titik operasi
> sistem yang sebenarnya adalah `FusionConfig.mild_at / severe_at / critical_at`,
> yang bisa diatur dari sidebar Streamlit tanpa restart. Ambang classifier
> mengatur angka yang dilaporkan `scripts/evaluate_fatigue.py` dan pemakaian
> model secara berdiri sendiri.

### Kalibrasi probabilitas

Karena fusi memakai probabilitas mentah, yang penting bukan ambangnya melainkan
apakah angka itu berarti. Diukur di test set:

| Ukuran | Nilai |
|---|---:|
| Expected Calibration Error (10 bin) | **0,0399** |
| Rata-rata p(fatigue) | 0,4848 |
| Proporsi label fatigue sebenarnya | 0,5000 |
| p rata-rata untuk kelas non-fatigue | 0,1256 |
| p rata-rata untuk kelas fatigue | 0,8440 |

Kalibrasinya baik — label smoothing 0,05 saat training membantu di sini. Tapi
ada sifat lain yang jauh lebih konsekuensial: **56,7% keluaran jatuh di bawah
0,05 atau di atas 0,95.** Distribusinya hampir biner, jadi kontribusi CNN ke
skor fusi praktis berupa saklar — nol, atau bobot penuh. Itu yang memaksa
perubahan bobot di bagian berikutnya.

### Perbandingan backend (test set, jalur inferensi sungguhan)

`scripts/evaluate_fatigue.py` mengevaluasi ulang lewat jalur yang benar-benar
dipakai runtime (`classifier.preprocess_bgr` dengan OpenCV), bukan jalur
`torchvision` yang dipakai training:

Semuanya pada ambang 0,2858 (hasil `recall-floor`):

| Backend | Akurasi | Presisi | Recall | F1 | ROC-AUC | ms/gambar¹ | Ukuran |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch | 0,8848 | 0,8432 | 0,9455 | 0,8914 | **0,9656** | 36,7 | 17,0 MB |
| OpenVINO FP32 | 0,8848 | 0,8432 | 0,9455 | 0,8914 | **0,9656** | 19,7 | 17,0 MB |
| OpenVINO INT8 | 0,8939 | 0,8611 | 0,9394 | 0,8986 | 0,9636 | **14,1** | **4,9 MB** |

¹ batch 1 — kasus realistis satu wajah per frame. Angka latensi berayun
±30% antar-run tergantung beban CPU lain; yang stabil adalah urutannya.

Tiga hal yang layak diperhatikan:

**Export OpenVINO FP32 lossless.** ROC-AUC-nya identik sampai empat desimal
dengan PyTorch, sambil ~2–3× lebih cepat. Tidak ada alasan menjalankan PyTorch
di produksi.

**ROC-AUC tidak berubah saat ambang diubah.** Itu memang seharusnya: AUC
mengukur kualitas *peringkat*, dan ambang tidak mengubah peringkat. Yang
berubah hanya di titik mana peringkat itu dipotong. Kalau suatu saat angka AUC
ikut bergerak setelah menyetel ambang, ada yang salah di pipeline evaluasinya.

**INT8: ongkosnya kecil, dan pada ambang ini justru sedikit menguntungkan.**
Model menyusut 3,5× (17,0 → 4,9 MB) dengan ROC-AUC turun 0,0020. Pada ambang
rendah 0,2858 ia kebetulan sedikit unggul di akurasi (0,8939 vs 0,8848) karena
kuantisasi membuatnya sedikit lebih konservatif — tapi jangan membaca itu
sebagai "INT8 lebih baik": AUC-nya tetap lebih rendah, dan keunggulan itu
artefak dari titik potong tertentu, bukan model yang lebih bagus. Pilih INT8
kalau ukuran dan kecepatan penting; FP32 kalau tidak.

Angka mentahnya ada di `outputs/fatigue/eval_<backend>.json`.

**Catatan metodologi:** akurasi test di tabel training (0,9182, pada ambang
Youden 0,5496) tidak sebanding langsung dengan tabel ini — ambangnya berbeda.
Pada ambang yang sama pun masih ada selisih ~3 gambar dari 330. Penyebabnya bukan
model, melainkan resampling: training mengevaluasi lewat `torchvision`/PIL,
sedangkan runtime memakai `cv2.resize`. Selisih itu diukur dan dibatasi oleh
`tests/test_fatigue_classifier.py`, dan angka yang benar untuk dipakai adalah
angka backend — itu yang akan dialami di lapangan.

---

## Cara sistem memutuskan level kelelahan

Skor CNN **tidak pernah** dipakai sendirian. Ia satu dari lima sumber bukti
yang diagregasi sepanjang jendela geser.

### Sinyal yang diukur

| Sinyal | Definisi | Kenapa dipakai |
|---|---|---|
| **PERCLOS** | Fraksi waktu kelopak mata tertutup dalam jendela | Ukuran mengantuk paling mapan (Wierwille, 1994); korelasi terkuat dengan penurunan performa |
| **Microsleep** | Penutupan mata terus-menerus ≥ 1,5 dtk | Satu kejadian saja sudah signifikan — beda kelas dari PERCLOS yang gradual |
| **Laju kedip** | Kedipan per menit | Naik saat lelah ringan, **turun** saat lelah berat (kedip jadi lebih lama, bukan lebih sering) |
| **Laju menguap** | Bukaan mulut bertahan ≥ 1,5 dtk, per menit | Ambang durasi memisahkan menguap dari bicara/tertawa |
| **Kepala terkulai** | Pitch ≥ 22° bertahan ≥ 1,2 dtk, per menit | Membedakan terkulai dari sekadar melihat ke bawah |
| **Skor CNN** | Probabilitas kelas fatigue, dirata-rata di jendela | Menangkap tanda yang tidak bergerak: mata sayu, lingkar hitam, ekspresi lemas |

Laju kedip diperlakukan khusus: hubungannya dengan kelelahan **tidak monoton**.
Laju di bawah 6/menit diperlakukan sebagai bukti lemah, bukan sebagai tanda
kesegaran — yang membedakan "jarang berkedip karena fokus" dari "jarang
berkedip karena hampir tertidur" adalah PERCLOS, dan itu ditimbang terpisah.

### Dua sumber untuk mata tertutup

Sengaja redundan:

- **Blendshape** `eyeBlinkLeft/Right` dari MediaPipe — sudah dinormalisasi
  antar-orang, tahan terhadap bentuk mata sipit/lebar. Sumber utama.
- **EAR geometris** dari koordinat landmark — absolutnya tidak sebanding
  antar-orang, tapi *perubahannya* relatif terhadap baseline orang itu sangat
  informatif, dan ia tetap ada saat blendshape gagal.

### Kalibrasi ambang per orang

Baseline EAR mata-terbuka dikumpulkan dari frame yang blendshape-nya jelas
menyatakan mata terbuka (`< 0.15`), sehingga kalibrasi tidak tercemar oleh
frame saat orangnya memang sedang memejamkan mata. Setelah 45 sampel, ambang
personal ditetapkan = median baseline × 0,72 (setara kriteria PERCLOS "P80").

Median, bukan rata-rata: satu frame buram dengan EAR ekstrem tidak boleh
menggeser ambang seseorang untuk selamanya.

Kalibrasi **bertahan** meski riwayatnya di-reset — bentuk mata seseorang tidak
berubah saat ia keluar-masuk frame, dan mengulang kalibrasi dari nol hanya
membuat sistem buta lagi selama beberapa detik.

### Struktur keputusan

Dua lapis, bukan satu skor tunggal:

**Skor tertimbang** (bobot default; bisa diatur di UI):

| Sinyal | Bobot | Jenuh di | Level tertinggi bila sendirian |
|---|---:|---|---|
| PERCLOS | 0,40 | 40% | **WASPADA** |
| CNN | 0,20 | 1,0 | SEGAR |
| Menguap | 0,20 | 4/menit | SEGAR |
| Kedip | 0,10 | 32/menit | SEGAR |
| Terkulai | 0,10 | 5/menit | SEGAR |

Tiap sinyal jenuh di titik tertentu, jadi kontribusi maksimum satu sumber
persis sama dengan bobotnya.

#### Invarian: hanya PERCLOS yang boleh berbicara sendirian

Kolom terakhir di tabel itu bukan kebetulan. Bobot CNN, menguap, kedip, dan
terkulai semuanya **di bawah `mild_at` (0,30)**, sehingga tidak satu pun bisa
menaikkan level tanpa dukungan sinyal lain.

Ini menutup masalah yang nyata, bukan hipotetis. Bobot CNN awalnya 0,30 —
persis sama dengan ambang WASPADA. Digabung dengan sifat distribusi
classifier yang hampir biner (56,7% keluarannya <0,05 atau >0,95), artinya
satu keluaran CNN yang keliru-tapi-yakin cukup untuk melaporkan orang yang
matanya terbuka lebar sepanjang menit itu sebagai waspada — dan tidak ada apa
pun di dalam sistem yang bisa membantahnya. Prinsip "CNN tidak pernah dipakai
sendirian" tertulis di mana-mana tapi tidak pernah benar-benar ditegakkan.

Sekarang ia ditegakkan, dan diuji
(`test_only_perclos_can_escalate_alone`). PERCLOS dikecualikan dengan sengaja:
ia satu-satunya pengukuran fisik langsung dengan definisi yang jelas, bukan
tebakan model, dan PERCLOS 40% memang layak jadi peringatan tanpa perlu
dikuatkan apa pun.

Bobot bisa diubah di sidebar Streamlit, termasuk ke setelan yang melonggarkan
invarian ini (preset "Fokus tampilan wajah" sengaja begitu, untuk kamera yang
sudut matanya buruk sehingga PERCLOS tidak andal). Ketika itu terjadi, UI
mengatakannya di tempat — `FusionConfig.sources_that_can_escalate_alone()`
dihitung ulang tiap kali slider digeser, jadi konsekuensinya terlihat saat
setelan diubah, bukan ditemukan lewat alarm palsu seminggu kemudian.

**Aturan keras** — satu kejadian yang sudah cukup bukti dengan sendirinya
langsung mengangkat level, berapa pun skor lunaknya:

- Microsleep terdeteksi → minimal **LELAH**
- Mata terpejam ≥ 3 detik → **KRITIS**

Menunggu rata-rata jendela 60 detik naik untuk kejadian seperti itu berarti
peringatan datang puluhan detik setelah orangnya tertidur.

### Histeresis

Level **naik seketika**, tapi hanya boleh **turun** setelah kondisinya membaik
selama 20 detik berturut-turut (bisa diatur).

Asimetri ini disengaja. Menunda peringatan saat kondisi memburuk adalah
kesalahan yang biayanya ditanggung orang di lapangan. Sedangkan level yang
berkedip-kedip antara LELAH dan SEGAR membuat operator berhenti mempercayai
dashboard-nya — dan dashboard yang tidak dipercaya sama tidak bergunanya dengan
dashboard yang mati.

### Empat level + satu keadaan "tidak tahu"

| Level | Arti |
|---|---|
| `TIDAK_DIKETAHUI` | Wajah belum cukup lama terlihat jelas (< 5 dtk pengamatan, atau < 40% frame menghasilkan landmark) |
| `SEGAR` | Tidak ada tanda kelelahan menonjol |
| `WASPADA` | Tanda awal — peringatan dini, belum perlu tindakan |
| `LELAH` | Bukti jelas, perlu tindakan |
| `KRITIS` | Berbahaya — microsleep panjang atau bukti menumpuk |

`TIDAK_DIKETAHUI` bukan hiasan. Orang yang membelakangi kamera menghasilkan
banyak frame tak-terpakai; melaporkan "SEGAR" untuk mereka sama menyesatkannya
dengan melaporkan "LELAH". Sistem ini menolak menebak.

### Setiap keputusan disertai alasan

```
Budi Santoso: LELAH
  — PERCLOS 34% (mata tertutup 34% waktu)
  — 2x microsleep (terlama 2,3 dtk)
  — menguap 1,8x/menit
```

Alert tanpa alasan akan diabaikan atau dimatikan. Penjelasan ini bagian dari
sistemnya, bukan hiasan.

---

## Absensi

### Model penyimpanan

SQLite satu file (`data/attendance.db`), tiga tabel: `employees`,
`embeddings`, `attendance`. Sistem ini dirancang jalan offline di mesin edge
dekat CCTV; menambah server database untuk data sebesar "beberapa ratus
karyawan × satu vektor 128 float" hanya menambah hal yang bisa mati saat pabrik
sedang berjalan. Backup cukup menyalin satu file.

### Banyak foto per orang, skor diambil yang tertinggi

Satu vektor hanya mewakili satu pose dan satu pencahayaan. Merata-ratakan foto
yang beragam justru menghasilkan vektor yang tidak mirip pose mana pun — dan
menghukum orang yang mendaftar paling lengkap.

Karena itu semua foto pendaftaran disimpan utuh, dan skor seseorang =
similarity **tertinggi** di antaranya. Mendaftar 5–10 foto dari sudut dan
cahaya berbeda membuat pengenalan jauh lebih tahan kondisi lapangan.

**Kualitas pendaftaran berpengaruh jauh lebih besar pada keandalan absensi
daripada model yang dipakai.** Foto **ditolak** kalau wajahnya tidak
terdeteksi, ada lebih dari satu wajah, wajahnya lebih kecil dari 80 px, atau
nyaris identik dengan foto yang sudah diterima — dan alasannya selalu
disebutkan. Lebih baik ditolak sekarang daripada jadi karyawan yang "kadang
tidak terbaca" berbulan-bulan.

Foto yang agak buram hanya **diperingatkan**, tidak diblokir. Gerbang blur
awalnya menolak (variance Laplacian < 40) dan itu keliru: sebuah frame webcam
640×480 dengan wajah frontal dan jelas — yang terbukti dikenali dengan
sempurna — hanya mendapat nilai 17, sementara foto internet terkurasi di
dataset mendapat 90–1600. Metrik itu lebih banyak mengukur *asal* gambar
(sensor webcam yang lembut vs foto yang sudah dipertajam) daripada
kelayakannya. Diuji dengan blur Gaussian bertingkat pada foto yang sama,
similarity embedding masih 0,62 pada kernel 21×21 — jauh di atas ambang
pengenalan 0,40 — padahal ketajamannya sudah runtuh ke 3,1. Orang yang gagal
mendaftar sama sekali tidak akan pernah dikenali kamera, jadi gerbang yang
terlalu ketat lebih merugikan daripada foto yang kurang ideal.

Metriknya juga diperbaiki: crop wajah dinormalisasi ke ukuran tetap sebelum
diukur. Tanpa itu, nilainya justru **naik** saat wajah mengecil (tepi jadi
lebih tajam relatif terhadap piksel) — sehingga wajah yang jauh dari kamera
tampak lebih tajam daripada wajah yang sama dari dekat, dan ambang apa pun
jadi bergantung pada jarak. Terukur: setelah normalisasi, nilai untuk citra
yang sama di 400/240/120 px identik; tanpa normalisasi ia bervariasi 3,7×.

Ketiga jalur pendaftaran — CLI, UI Streamlit, dan endpoint API — memanggil
aturan yang sama di [`enrollment.py`](../src/fatigue/enrollment.py).
Sebelumnya masing-masing punya aturannya sendiri dan ketiganya berbeda,
sehingga karyawan yang didaftarkan lewat UI diam-diam mendapat validasi yang
lebih longgar tanpa ada yang memberitahunya. Ada test yang memastikan
duplikasi itu tidak tumbuh lagi.

### Ambang pengenalan

Default 0.40 untuk SFace (rekomendasi OpenCV Zoo 0.363, dinaikkan). Absensi
lebih menderita akibat salah-orang (*false accept*) daripada akibat diminta
menghadap kamera sekali lagi.

Embedding dari backend berbeda **tidak sebanding**. Mengganti
`EMBEDDER_BACKEND` menuntut pendaftaran ulang seluruh karyawan; database
menolak mencampur dimensi vektor dan mengatakan alasannya.

### State fatigue diikat ke identitas, bukan ke kotak wajah

Tracker berbasis IoU biasa kehilangan riwayat seseorang setiap kali ia
keluar-masuk frame atau tertukar dengan rekan di sebelahnya — dan riwayat 60
detik yang hilang berarti sistemnya buta lagi selama satu menit. Karena
wajahnya toh sudah di-embed untuk absensi, mengikat state ke identitas hasil
pengenalan itu gratis dan jauh lebih stabil.

Orang yang belum terdaftar tetap dilacak dengan identitas sementara berbasis
kemiripan embedding antar-frame. Dan kalau ia kemudian **didaftarkan di tengah
sesi**, riwayat matanya dipindahkan ke track karyawan, tidak dibuang.

### Cooldown log

Tanpa cooldown, satu orang yang berdiri di depan kamera 10 detik menghasilkan
250 baris log. Cooldown default 5 menit, **per orang** — dua karyawan yang
lewat bersamaan tetap dua-duanya tercatat.

Hanya frame yang benar-benar menjalankan pengenalan yang boleh membuat baris
absensi. Frame yang cuma dicocokkan lewat IoU mewarisi nama, dan mewarisi nama
bukan bukti kehadiran.

---

## Pemakaian

### CLI

```bash
# realtime dari webcam/CCTV
python -m src.fatigue.cli webcam
python -m src.fatigue.cli webcam --index 1 --save outputs/shift_pagi.mp4

# analisis rekaman (waktu diambil dari posisi frame, bukan jam dinding)
python -m src.fatigue.cli video rekaman.mp4 --stride 2

# cek satu gambar (hanya wajah + identitas; level fatigue mustahil dari 1 frame)
python -m src.fatigue.cli image foto.jpg

# tanpa absensi (hemat 37 MB) atau tanpa CNN (sinyal perilaku saja)
python -m src.fatigue.cli --no-attendance webcam
python -m src.fatigue.cli --no-classifier webcam
```

### Streamlit

```bash
streamlit run app/streamlit_app.py
```

Pilih **Fatigue & absensi** di sidebar. Tiga tampilan: **Monitor** (webcam /
video / gambar), **Karyawan** (daftar + pendaftaran + penghapusan), **Log
absensi** (riwayat + unduh CSV).

### FastAPI

```bash
uvicorn app.api:app --reload
```

| Endpoint | Kegunaan |
|---|---|
| `GET /fatigue/health` | Status komponen + statistik absensi |
| `POST /fatigue/analyze?session_id=cam1` | Analisis satu frame dalam konteks sesi |
| `POST /fatigue/session/{id}/reset` | Buang riwayat temporal satu sesi |
| `GET /fatigue/sessions` | Sesi aktif |
| `GET /fatigue/employees` | Daftar karyawan |
| `POST /fatigue/employees` | Daftarkan karyawan (multipart, banyak foto) |
| `DELETE /fatigue/employees/{id}` | Hapus permanen |
| `GET /fatigue/attendance` | Log kehadiran |

**Penting:** kirim frame berurutan dengan `session_id` yang sama. Riwayat
temporal terakumulasi di server per sesi. Request tanpa session yang konsisten
akan selalu mengembalikan `TIDAK_DIKETAHUI` — itu jujur, bukan bug.

Sesi menua sendiri setelah 10 menit idle (maks. 32 sesi).

---

## Konfigurasi

Semua lewat `.env` (lihat `.env.example`):

| Variabel | Default | Keterangan |
|---|---|---|
| `EMBEDDER_BACKEND` | `sface` | `sface` \| `insightface` |
| `FATIGUE_BACKEND` | `torch` | `torch` \| `openvino` \| `openvino-int8` |
| `FATIGUE_MODEL_PATH` | `models/fatigue/fatigue_cls.pt` | Checkpoint classifier |
| `FATIGUE_ASSET_DIR` | `models/fatigue` | Lokasi bobot pihak ketiga |
| `ATTENDANCE_DB` | `data/attendance.db` | Database absensi |
| `CAMERA_NAME` | — | Label kamera di log absensi |
| `FATIGUE_ATTENDANCE` | `1` | `0` = matikan absensi |
| `FATIGUE_CLASSIFIER` | `1` | `0` = matikan CNN |
| `ENABLE_FATIGUE_API` | `1` | `0` = jangan pasang endpoint `/fatigue/*` |

Ambang, bobot fusi, dan panjang jendela bisa diatur langsung dari sidebar
Streamlit tanpa restart.

---

## Performa

```bash
python scripts/benchmark_fatigue.py --source frame.jpg
python scripts/benchmark_fatigue.py --source frame.jpg --faces 4
python scripts/benchmark_fatigue.py --source frame.jpg --backend openvino-int8
```

Hasilnya ditulis ke `outputs/fatigue/benchmark*.json`, terurai per komponen —
bukan cuma total, karena keputusan optimasi tergantung pada mana yang dominan.

### Angka terukur

Frame 640×480, satu wajah, CPU Intel Core i7-1265U (10 core / 12 thread),
backend `openvino-int8`:

| Komponen | Median | Catatan |
|---|---:|---|
| Deteksi wajah (YuNet) | 66 ms | dominan; lihat penskalaan di bawah |
| Landmark (MediaPipe) | 37 ms | per wajah, tiap frame |
| Embedding (SFace) | 47 ms | per wajah, tapi 1:10 frame |
| Classifier INT8 | 23 ms | per wajah, tapi 1:5 frame |
| Classifier PyTorch | 77 ms | pembanding |

| Skenario | Latensi | FPS | Kamera/mesin² |
|---|---:|---:|---:|
| 1 wajah, CNN tiap frame, PyTorch | 222 ms | 4,5 | 0,6 |
| 1 wajah, CNN tiap frame, INT8 | 144 ms | 7,0 | 0,9 |
| 1 wajah, CNN 1:5, PyTorch | 133 ms | 7,5 | 0,9 |
| **1 wajah, CNN 1:5, INT8** | **112 ms** | **8,9** | **1,1** |
| 4 wajah, CNN 1:5, INT8 | 409 ms | 2,5 | 0,3 |

² dengan asumsi 8 fps per kamera.

**Ini laptop, bukan server.** Satu i7-1265U menangani kira-kira satu kamera
satu-orang secara realtime. Untuk multi-orang, biayanya tumbuh hampir linier
karena landmarker dan embedder jalan per wajah — 4 wajah ≈ 2,5 fps. Rencanakan
kapasitas dari angka per-wajah, bukan dari angka satu-wajah.

### Deteksi dijalankan pada frame yang diperkecil

YuNet adalah komponen termahal dan biayanya tumbuh dengan jumlah piksel. Karena
itu deteksi berjalan pada frame yang diperkecil ke `detect_width` (default 640)
lalu koordinatnya diskalakan kembali; landmarker dan embedder tetap bekerja
pada resolusi penuh.

Diukur pada frame 1920×1440:

| `detect_width` | Latensi deteksi | Similarity embedding vs resolusi penuh |
|---|---:|---:|
| `None` (penuh) | 296 ms | 1,0000 |
| **640** (default) | **45 ms** | **0,9861** |
| 320 | 20 ms | 0,9715 |

**6,5× lebih cepat** dengan similarity 0,986 — jauh di atas ambang identitas
absensi 0,40, jadi pengenalan tidak terpengaruh sama sekali. Yang berkurang
hanya kemampuan menemukan wajah yang sangat kecil, dan `min_face` ikut
diskalakan supaya ambangnya tetap dinyatakan dalam piksel frame asli.

Dua jalur sengaja TIDAK memakai penskalaan ini: `prepare_fatigue_dataset.py`
dan `enroll_faces.py`. Keduanya offline dan dijalankan sekali, dan pada
keduanya crop yang tepat lebih berharga daripada kecepatan.

### Optimasi lain yang sudah terpasang

**Embedding di-subsample.** Identitas seseorang tidak berubah antara dua frame.
Di sela-selanya wajah dicocokkan ke track lewat IoU yang ongkosnya nol.
Embedding tetap dihitung untuk wajah yang **baru muncul**, berapa pun nomor
frame-nya — kalau tidak, orang yang baru masuk ruangan tercatat "tidak dikenal"
sampai penyegaran berikutnya dan absensinya bisa terlewat.

**CNN di-subsample (1:5).** Penampakan wajah berubah jauh lebih lambat daripada
25 fps. 1 dari 5 frame memberi ~5 pembaruan/detik — jauh lebih rapat daripada
yang dibutuhkan jendela 60 detik. (Dengan OpenVINO, CNN sudah bukan komponen
termahal; menaikkan interval ini lebih jauh tidak banyak menolong. Yang dominan
sekarang deteksi wajah dan landmarker.)

**Landmark TIDAK di-subsample.** PERCLOS dan deteksi kedipan butuh resolusi
waktu penuh; menghemat di sini langsung merusak sinyal utamanya. Ia komponen
kedua termahal, dan itu memang harga yang dibayar untuk sinyal yang paling
dapat dipercaya di sistem ini.

FPS pemrosesan bukan FPS kamera. CCTV 25 fps tidak menuntut analisis 25 fps —
PERCLOS dan microsleep tetap terukur benar pada 8–10 fps, dan menurunkannya
adalah cara paling murah memuat lebih banyak kamera di satu mesin.

Untuk deploy, export ke OpenVINO:

```bash
python scripts/export_fatigue.py
# lalu FATIGUE_BACKEND=openvino        (akurasi identik PyTorch, 3,3x lebih cepat)
# atau FATIGUE_BACKEND=openvino-int8   (3,5x lebih kecil, recall sedikit turun)
```

FP32 adalah default yang disarankan: akurasinya identik dengan PyTorch dan
selisih kecepatannya terhadap INT8 kecil (12,3 vs 10,4 ms). Pilih INT8 kalau
ukuran image atau memori edge box benar-benar menjadi kendala.

---

## Batasan yang perlu diketahui

Ditulis di sini karena mengetahuinya lebih berguna daripada angka akurasi mana
pun.

**Dataset trainingnya foto internet, bukan frame CCTV pabrik.** Label "lelah"
di dataset ini adalah penilaian visual pihak ketiga terhadap foto, bukan
kondisi kelelahan yang terukur (PVT, KSS, EEG). Model belajar "wajah yang
*terlihat* lelah menurut penilai", dan itu tidak sama dengan "orang yang
*sedang* lelah". Untuk deployment serius, kumpulkan data dari kamera dan
pencahayaan Anda sendiri, lalu latih ulang — script-nya sudah siap menerima
folder `fatigue/` dan `nonfatigue/` mana pun.

**Sinyal temporalnya jauh lebih dapat dipercaya daripada CNN-nya.** PERCLOS dan
microsleep adalah pengukuran fisik yang definisinya jelas dan tidak bergantung
pada label siapa pun. Kalau harus memilih satu, matikan CNN
(`--no-classifier`) dan percayai perilakunya.

**Kacamata, terutama yang memantul, mengganggu landmark mata.** Masker menutupi
sinyal menguap. Helm yang turun rendah bisa memotong dahi. Semuanya menurunkan
`usable_ratio`, dan sistem akan melaporkan `TIDAK_DIKETAHUI` alih-alih menebak
— tapi itu berarti orang tersebut tidak terpantau.

**Wajah harus ≥ 40 px.** Di bawah itu tidak tersisa cukup piksel mata untuk
PERCLOS. Untuk kamera yang jauh, ini berarti perlu lensa yang lebih panjang
atau resolusi yang lebih tinggi — bukan ambang yang diturunkan.

**Biayanya tumbuh per wajah, bukan per frame.** Landmarker dan embedder jalan
sekali untuk setiap orang di frame. Satu laptop i7 menangani satu kamera
satu-orang pada ~9 fps, tapi jatuh ke ~2,5 fps untuk empat orang. Untuk area
ramai, rencanakan satu mesin per kamera atau turunkan `max_faces` dan terima
bahwa sebagian orang tidak terpantau — jangan berasumsi satu server melayani
belasan titik.

**Ini bukan alat diagnosis medis.** Keluarannya adalah indikator untuk
penjadwalan istirahat dan pengawasan, bukan penilaian kelayakan kerja seseorang.

---

## Privasi & kepatuhan

Data wajah adalah data biometrik dan diperlakukan sebagai data pribadi.

- Yang disimpan adalah **vektor embedding, bukan foto wajah**. Vektor itu tetap
  data biometrik.
- `data/` dan `*.db` ada di `.gitignore`. Database absensi **tidak boleh**
  masuk repositori.
- `delete_employee` menghapus karyawan berikut seluruh embedding dan log-nya
  secara permanen (`ON DELETE CASCADE`, dengan `PRAGMA foreign_keys = ON`).
  Kemampuan menghapus ini wajib ada, bukan opsional.
- `set_active(False)` menonaktifkan tanpa menghapus, untuk karyawan cuti atau
  pindah shift.
- Semua pemrosesan terjadi di mesin lokal. Tidak ada frame, embedding, atau
  identitas yang dikirim ke mana pun.

Sebelum dipakai di lingkungan kerja nyata: pastikan ada dasar hukum dan
persetujuan yang sesuai (di Indonesia: UU PDP No. 27/2022, yang mengategorikan
data biometrik sebagai data pribadi spesifik), pemberitahuan yang jelas kepada
karyawan, dan kebijakan retensi. Kemampuan teknis sistem ini tidak
menggantikan kewajiban itu.

---

## Struktur file

```
src/fatigue/
    types.py        dataclass bersama (murni data, tanpa torch/cv2/mediapipe)
    assets.py       unduh + verifikasi sha256 bobot pihak ketiga
    face.py         YuNet + embedder pluggable (SFace / InsightFace)
    landmarks.py    MediaPipe FaceLandmarker -> EAR, MAR, pose kepala
    classifier.py   CNN fatigue + preprocessing bersama (torch / OpenVINO)
    temporal.py     jendela geser: PERCLOS, kedip, menguap, microsleep, kalibrasi
    fusion.py       skor tertimbang + aturan keras + histeresis + penjelasan
    attendance.py   SQLite: karyawan, embedding, log kehadiran
    enrollment.py   validasi & pendaftaran foto — dipakai CLI, UI, dan API
    pipeline.py     orkestrasi + asosiasi track + rendering
    cli.py          CLI image/video/webcam

scripts/
    prepare_fatigue_dataset.py   unduh Kaggle + crop wajah + split per identitas
    train_fatigue.py             transfer learning + EMA + tuning ambang
    evaluate_fatigue.py          metrik test set per backend + sapuan ambang
    export_fatigue.py            ONNX + OpenVINO FP32/INT8
    tune_fatigue_threshold.py    pilih titik operasi classifier, tanpa training ulang
    benchmark_fatigue.py         latensi per komponen + FPS ujung-ke-ujung
    enroll_faces.py              pendaftaran wajah (folder / webcam)

app/
    fatigue_api.py  router FastAPI /fatigue/*
    fatigue_ui.py   halaman Streamlit

tests/
    test_fatigue_temporal.py     PERCLOS, microsleep, kalibrasi, fusi, histeresis
    test_fatigue_attendance.py   pendaftaran, pencocokan, cooldown, penghapusan
    test_fatigue_pipeline.py     asosiasi track, subsampling, integrasi absensi
    test_fatigue_face.py         penskalaan deteksi, filter ukuran, embedding
    test_fatigue_classifier.py   kesetaraan preprocessing training vs inferensi
    test_fatigue_enrollment.py   aturan penerimaan foto, dan bahwa ketiganya sama

120 test, semuanya jalan tanpa GPU. Yang butuh bobot atau checkpoint akan
di-skip, bukan gagal, kalau file-nya belum ada.
```
