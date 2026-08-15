# Draft Caption LinkedIn — PPE Detection Project

> Bahasa Indonesia · Siap posting · Ganti `@INSTRUKTUR` dan `@AI_SUPER_CLASS`
> dengan handle sesuai instruksi tugas kelas.

- **Post #1** (Versi A & B) — rilis awal: YOLOv8 + FastAPI + Streamlit + OpenVINO.
- **Post #2** (Versi C & D) — iterasi UI operator: sensitivitas per kategori,
  kebijakan alarm, log bukti. Ada di bagian bawah file ini.
- **Post #3** (Versi E & F) — versi tugas akhir: hasil training apa adanya dan
  cara membacanya per kelas.
- **Post #4** (Versi G & H) — demo web yang bisa diklik: trade-off payload,
  backpressure, dan bug yang ketahuan lewat tes piksel.

---

# Post #1 — Rilis awal

## Versi A — Storytelling (± 260 kata, cocok untuk post utama)

🦺 **Dari deteksi satu-frame ke sistem PPE Detection yang siap deploy.**

Beberapa bulan lalu saya bikin proyek kecil: script Python yang manggil
Roboflow Serverless API untuk deteksi APD dari webcam. Jalan, tapi berhenti
di situ — belum jadi "produk". Sebagai tugas akhir sertifikasi **AI Super
Class (Computer Vision Track)** dari @AI_SUPER_CLASS, saya lanjutkan proyek
itu menjadi sistem end-to-end.

Yang saya bangun:
🔹 **Inference lokal** pakai YOLOv8 (Ultralytics), tidak lagi bergantung API online
🔹 **REST API** dengan FastAPI — `/predict` menerima upload gambar, kembalikan JSON deteksi + gambar teranotasi
🔹 **Demo UI** dengan Streamlit — upload foto/video/webcam realtime, plus kartu compliance status per kelas APD
🔹 **Containerized** dengan Docker Compose (FastAPI + Streamlit jalan bareng)
🔹 Dataset 22.732 gambar, **17 kelas** yang dipetakan jadi **8 kategori kepatuhan** (helm, kacamata, masker, sarung tangan, sepatu, rompi, pelindung telinga, harness) — tiap kategori punya label positif & label pelanggaran
🔹 **Optimasi OpenVINO** — model di-export ke INT8 dan dijalankan di iGPU Intel

Bagian yang paling banyak mengajari saya justru yang terakhir. Model PyTorch
di CPU cuma dapat **6,4 FPS** — tidak cukup untuk stream kamera. Setelah
export ke OpenVINO INT8 dan dijalankan di **iGPU Intel yang selama ini
menganggur**, hasilnya **54,9 FPS — 8,55× lebih cepat**, dengan ukuran model
turun dari 11,7 MB ke 3,4 MB.

Pertanyaan wajibnya: berapa akurasi yang dikorbankan? Saya ukur di split test
yang belum pernah dilihat model — **cuma turun 0,32 poin mAP@50**
(65,45% → 65,13%). Untuk deteksi APD trade-off ini jelas menguntungkan:
*melewatkan* pekerja tanpa helm jauh lebih mahal daripada satu false positive
yang bisa diverifikasi supervisor.

Yang bikin proyek ini "nyambung" ke pekerjaan saya sehari-hari: saya kerja di
industri **fleet GPS & dashcam**. Kepatuhan APD driver/kernet adalah concern
nyata di sisi HSE armada. Dan angka 8,55× tadi bukan sekadar benchmark —
itulah yang menentukan apakah deteksi bisa jalan di **edge box murah di
lapangan**, atau harus streaming semua video ke cloud (mahal, dan sering
mustahil di area tambang dengan sinyal seadanya).

CV bukan cuma soal model — bagian tersulit justru bikin pipeline yang
reliable, gampang dipakai, dan **muat di perangkat yang benar-benar ada di
lapangan**. Sertifikasi ini memaksa saya untuk *ship* seluruh alur itu, bukan
berhenti di notebook.

Terima kasih banyak untuk mentoring @INSTRUKTUR dan tim @AI_SUPER_CLASS 🙏

Repo & demo di komentar 👇

#ComputerVision #YOLOv8 #Ultralytics #OpenVINO #EdgeAI #FastAPI #Streamlit
#Docker #PPEDetection #AIforSafety #FleetManagement #Dashcam #K3
#AISuperClass #MachineLearning #DeepLearning

---

## Versi B — Ringkas (± 120 kata, cocok kalau mau padat)

🦺 Selesai sudah tugas akhir **AI Super Class — CV Track** dari
@AI_SUPER_CLASS: PPE Detection end-to-end (YOLOv8 + FastAPI + Streamlit +
Docker).

Ini kelanjutan dari proyek kecil saya sebelumnya yang cuma panggil API
serverless — sekarang naik jadi sistem lokal, punya REST API, punya UI demo,
dan siap di-containerize. Dataset 22.732 gambar, 17 kelas, dipetakan jadi 8
kategori kepatuhan APD.

Highlight buat saya: optimasi **OpenVINO INT8** di iGPU Intel menaikkan
inference dari **6,4 FPS → 54,9 FPS (8,55×)** dengan biaya akurasi cuma
**0,32 poin mAP@50**. Itu selisih antara "harus streaming ke cloud" dan
"cukup jalan di edge box lapangan".

Kenapa relevan buat saya: sehari-hari saya kerja di **fleet GPS & dashcam**.
Deteksi kepatuhan APD driver via dashcam / CCTV site adalah use case nyata
yang bisa langsung diadopsi. Next step: integrasi ke RTSP stream armada dan
event pipeline ke dashboard HSE.

Terima kasih mentoring dari @INSTRUKTUR 🙏

#ComputerVision #YOLOv8 #OpenVINO #EdgeAI #FastAPI #Streamlit #PPEDetection
#Dashcam #FleetManagement #AISuperClass #K3

---

## Catatan pemakaian (post #1)

- **Tag instruktur:** ganti `@INSTRUKTUR` dan `@AI_SUPER_CLASS` dengan
  handle LinkedIn resmi yang diminta pihak kelas.
- **Repo/demo di komentar:** siapkan link GitHub repo, screenshot, atau
  short-clip demo (max 30 detik) dan taruh di komentar pertama supaya
  post utama tetap ringkas.
- **Waktu posting:** LinkedIn engagement Indonesia biasanya bagus di jam
  08.00–10.00 atau 19.00–21.00 WIB.
- **Media:** post carousel 3–5 slide (arsitektur diagram, screenshot UI,
  screenshot Swagger, GIF deteksi) menaikkan reach signifikan.

---

# Post #2 — Iterasi UI operator

> Sudut pandang yang dijual di sini **bukan** akurasi model (itu sudah dipakai
> di post #1), tapi jarak antara "model jalan" dan "sistem yang mau dipakai
> orang". Jangan ulang angka 8,55× sebagai headline — cukup disinggung.

## Versi C — Storytelling (± 330 kata, post utama)

🦺 **Model yang akurat belum tentu jadi sistem yang dipakai.**

Bulan lalu saya posting PPE Detection: YOLOv8 dioptimasi OpenVINO, jalan
realtime di iGPU. Minggu ini saya lanjutkan — tapi bukan dengan menaikkan
mAP. Saya duduk dan membayangkan satu orang: petugas HSE yang harus menatap
layar itu delapan jam sehari.

Begitu dibayangkan begitu, masalahnya ternyata bukan akurasi:

⚠️ **Alarm yang tidak dipercaya.** Deteksi per-frame itu berkedip. Pekerja
menoleh sebentar, frame sedikit buram, dan status "tanpa helm" muncul-hilang
belasan kali dalam semenit. Alarm yang bunyi 200 kali sejam pasti diabaikan —
dan sistem yang diabaikan sama saja dengan tidak ada.
→ Pelanggaran baru dialarmkan setelah bertahan N frame berturut-turut, plus
cooldown per kategori.

🎚️ **Satu ambang untuk semua kelas itu keliru.** Di model saya `boots` dapat
mAP 0,90 sementara `hand_noglove` cuma 0,26 — masuk akal, "tangan tanpa
sarung tangan" secara visual ya cuma tangan biasa. Menurunkan ambang global
demi kelas lemah bikin semua kelas lain ikut berisik.
→ Ambang confidence bisa diatur **per kategori**.

📋 **Alarm tanpa bukti tidak bisa ditindaklanjuti.** → Log pelanggaran yang
bisa diunduh CSV, snapshot bukti otomatis, dan rekaman sesi.

🏭 **Tidak semua area kerja punya aturan sama.** → Kategori mana yang
dideteksi dan mana yang boleh membunyikan alarm bisa dipilih terpisah, lalu
disimpan sebagai profil per area.

Detail kecil yang paling berkesan buat saya: rekaman bukti awalnya saya tulis
di 15 FPS, karena angka itu "kelihatan wajar". Padahal inference cuma sanggup
sekitar 5 FPS — jadi videonya 3× lebih cepat dari kejadian aslinya. Untuk
konten hiburan tidak masalah; untuk bukti kepatuhan, durasi yang salah bikin
rekaman itu tidak layak dipakai. Sekarang FPS-nya diukur dulu, baru file-nya
dibuat.

Pelajaran yang saya bawa: setelah model jadi, sebagian besar pekerjaan yang
tersisa bukan lagi soal ML. Itu soal desain produk — dan kesabaran menulis 51
unit test untuk logika yang kelihatannya sepele.

Sehari-hari saya kerja di **fleet GPS & dashcam**, dan polanya persis sama
dengan alarm telematics: yang menentukan sebuah fitur dipakai atau justru
dimatikan bukan akurasinya, tapi berapa kali dia salah membangunkan orang.

Repo di komentar 👇

#ComputerVision #YOLOv8 #OpenVINO #EdgeAI #Streamlit #PPEDetection #K3 #HSE
#FleetManagement #Dashcam #ProductThinking #MachineLearning

## Versi D — Ringkas (± 130 kata)

🦺 Update PPE Detection saya — dan kali ini nyaris tidak menyentuh model.

Setelah dicoba seperti orang yang benar-benar harus memakainya seharian,
masalahnya bukan akurasi, tapi **alarm fatigue**: deteksi per-frame berkedip,
alarm bunyi puluhan kali semenit, lalu diabaikan.

Yang saya tambahkan:
🔹 Debounce — pelanggaran harus bertahan N frame berturut-turut
🔹 Cooldown per kategori
🔹 Ambang confidence **per kategori** (di model saya `boots` mAP 0,90 vs
`hand_noglove` 0,26 — tidak adil dipukul rata)
🔹 Pilih terpisah: apa yang **dideteksi** vs apa yang boleh **membunyikan alarm**
🔹 Log pelanggaran + CSV, snapshot bukti, rekaman sesi

Pelajarannya: setelah model jadi, sisa pekerjaannya bukan ML lagi — itu
desain produk.

Repo di komentar 👇

#ComputerVision #YOLOv8 #OpenVINO #Streamlit #PPEDetection #K3 #HSE #EdgeAI

## Saran lampiran (post #2)

**Format terbaik: carousel PDF 5–6 slide.** LinkedIn memberi reach lebih
besar ke dokumen daripada gambar tunggal, dan ceritanya memang bertahap.

| Slide | Isi | Sumber |
|-------|-----|--------|
| 1 | Hook: "200 alarm sejam = 0 alarm yang dipercaya" di atas screenshot UI dengan banner pelanggaran merah | screenshot mode Webcam realtime |
| 2 | Masalah kedipan: potongan log/frame yang menunjukkan status berganti-ganti, vs hasil setelah debounce | bisa dibuat dari log CSV |
| 3 | Panel sidebar sensitivitas per kategori (slider terlihat jelas) | screenshot sidebar |
| 4 | Kontras mAP per kelas: `boots` 0,90 vs `hand_noglove` 0,26 — alasan ambang per kategori | `runs/ppe/ppe-yolov8n-cpu/` |
| 5 | Log pelanggaran + snapshot bukti + tombol unduh CSV | screenshot area utama |
| 6 | CTA: repo GitHub + "next: RTSP stream & event pipeline ke dashboard HSE" | — |

**Alternatif video (15–30 detik)** — engagement biasanya paling tinggi, dan
proyek ini punya keunggulan: fitur rekam sesi sudah ada, jadi tinggal pakai.
Alur yang enak ditonton: kamera menyala → helm dilepas → status berubah
merah → alarm berbunyi sekali (bukan spam) → snapshot bukti muncul di log.

**Hal yang wajib diperhatikan:**

- **Privasi.** Jangan tampilkan wajah rekan kerja, klien, atau lokasi kerja
  nyata tanpa izin tertulis. Paling aman: pakai diri sendiri, atau gambar
  dari dataset publik Roboflow yang memang dilisensikan untuk itu.
- **Video LinkedIn autoplay tanpa suara** — beri teks/caption di dalam video,
  jangan mengandalkan bunyi alarm sebagai puncak cerita.
- **Link repo taruh di komentar pertama**, bukan di badan post. LinkedIn
  menekan jangkauan post yang mengandung link keluar.
- **Jangan sebut demo online** sampai benar-benar ter-deploy.
- Jam posting yang bagus untuk audiens Indonesia: 08.00–10.00 atau
  19.00–21.00 WIB, hari Selasa–Kamis.

---

# Post #3 — Versi tugas akhir (menonjolkan hasil training di rubythalib.ai)

> Dipakai kalau post ini memang bagian dari submission tugas akhir. Beda
> penekanan dengan post #2: yang dijual di sini **hasil training dan cara
> membacanya**, bukan desain UI. Ganti `@Ruby Thalib` dengan tag LinkedIn
> yang benar sebelum posting.

## Versi E — Storytelling (± 350 kata, post utama)

🎓 **Tugas akhir saya di AI Super Class — Computer Vision Track
(rubythalib.ai): PPE Detection end-to-end.**

Yang paling berharga dari kelas ini ternyata bukan "cara melatih YOLO" —
tapi cara membaca hasilnya.

📊 **Hasil training saya, apa adanya.**
Dataset Roboflow 22.732 gambar, 17 kelas yang saya petakan jadi 8 kategori
kepatuhan APD. Model dilatih di subset **2.500 gambar, 416 px, 20 epoch, di
CPU laptop (i7-1265U), sekitar 4 jam**.

Hasil di split test — 948 gambar yang belum pernah dilihat model:
• mAP@50 — **65,5%**
• mAP@50-95 — **38,9%**
• Precision **67,6%** · Recall **62,8%**

Sebagai pembanding, arsitektur yang sama dilatih Roboflow di **dataset penuh
dengan GPU** mendapat mAP@50 **86,3%**.

Dulu saya akan berhenti di situ dan merasa gagal. Yang diajarkan kelas ini:
satu angka tidak menjelaskan apa-apa. Begitu dibuka per kelas, polanya
langsung terbaca:

✅ `boots` 0,905 · `Barefoots` 0,905 · `person` 0,881 · `vest` 0,858
❌ `hand_noglove` 0,255 · `No_Ear-Protection` 0,310 · `No_Glasses` 0,411

Modelnya bukan "kurang pintar" — kelas negatifnya yang memang jauh lebih
sulit. "Tangan tanpa sarung tangan" secara visual ya cuma tangan biasa:
objeknya kecil, tidak punya ciri khas, dan porsinya di subset sedikit. Itu
masalah **data**, bukan masalah epoch. Menambah epoch tidak akan
menyelesaikannya.

Kesimpulan itu yang lalu mengubah desain sistemnya: ambang confidence saya
buat bisa diatur **per kategori**, bukan satu angka untuk semua kelas. Dan
prioritas berikutnya jadi jelas — menambah contoh untuk kelas-kelas lemah
tadi, bukan menambah waktu training.

Sisanya menyusul di atas fondasi itu: export **OpenVINO INT8** (6,4 → 54,9
FPS di iGPU Intel, akurasi turun cuma 0,32 poin mAP@50), REST API FastAPI,
UI Streamlit dengan kebijakan alarm, Docker Compose, dan 51 unit test.

Sehari-hari saya kerja di **fleet GPS & dashcam**, jadi deteksi kepatuhan APD
lewat dashcam/CCTV site adalah use case yang benar-benar ada di meja saya.

Terima kasih @Ruby Thalib dan tim **rubythalib.ai** — kelasnya memaksa saya
menuntaskan seluruh alur sampai bisa dijalankan orang lain, bukan berhenti
di notebook 🙏

Repo di komentar 👇

#ComputerVision #YOLOv8 #Ultralytics #OpenVINO #EdgeAI #FastAPI #Streamlit
#Docker #PPEDetection #K3 #HSE #RubyThalibAI #AISuperClass #MachineLearning
#DeepLearning

## Versi F — Ringkas (± 140 kata)

🎓 Tugas akhir **AI Super Class — Computer Vision Track (rubythalib.ai)**:
PPE Detection end-to-end.

Hasil training saya apa adanya — subset **2.500 gambar, 416 px, 20 epoch, 4
jam di CPU laptop**: mAP@50 **65,5%**, precision 67,6%, recall 62,8% di split
test 948 gambar. Dilatih penuh di GPU oleh Roboflow, arsitektur yang sama
dapat 86,3%.

Yang paling saya bawa pulang dari kelas ini bukan angkanya, tapi cara
membacanya. Per kelas: `boots` 0,905 tapi `hand_noglove` cuma 0,255 — karena
"tangan tanpa sarung tangan" secara visual ya cuma tangan biasa. Itu masalah
data, bukan masalah epoch.

Temuan itu langsung mengubah desain sistem: ambang confidence bisa diatur per
kategori, dan prioritas berikutnya adalah menambah data kelas lemah — bukan
menambah training.

Sisanya: OpenVINO INT8 (6,4 → 54,9 FPS, -0,32 poin mAP), FastAPI, Streamlit,
Docker, 51 unit test.

Terima kasih @Ruby Thalib & tim rubythalib.ai 🙏

#ComputerVision #YOLOv8 #OpenVINO #EdgeAI #PPEDetection #RubyThalibAI #K3

## Saran lampiran (post #3)

Untuk post tugas akhir, lampiran terbaik adalah **bukti proses training** —
ini yang membedakan submission serius dari sekadar screenshot demo. Semua
file di bawah sudah ada di repo, tinggal dipakai.

| Slide | File | Kenapa |
|-------|------|--------|
| 1 | Screenshot UI dengan compliance card + judul tugas akhir | Pembuka yang langsung menunjukkan hasil akhir |
| 2 | `runs/ppe/ppe-yolov8n-cpu/results.png` | Kurva loss & mAP per epoch — bukti training benar-benar dijalankan sendiri |
| 3 | `runs/ppe/ppe-yolov8n-cpu/BoxPR_curve.png` | Precision-recall per kelas: inilah yang menunjukkan kelas kuat vs lemah |
| 4 | Tabel dari `docs/METRICS.md` (FP32 vs OpenVINO FP32 vs INT8) | Menunjukkan kuantisasi diukur, bukan diklaim |
| 5 | `runs/ppe/ppe-yolov8n-cpu/val_batch0_pred.jpg` | Grid prediksi di data validasi — visual paling meyakinkan |
| 6 | Tabel benchmark `docs/BENCHMARK.md` + CTA repo | Penutup: dari training sampai siap edge |

Catatan:

- `confusion_matrix.png` untuk 17 kelas biasanya terlalu padat dibaca di HP —
  pakai hanya kalau di-crop ke kelas yang dibahas.
- Sebutkan **hardware dan durasi training** (CPU i7-1265U, ~4 jam) di slide
  metrik. Justru itu yang membuat angka 65,5% masuk akal dan jujur.
- Kalau kelas mewajibkan tag/tema tertentu (nama batch, nomor angkatan,
  hashtag resmi), tambahkan di baris terakhir sebelum hashtag umum.

---

# Post #4 — Demo yang bisa diklik orang lain

> Angle yang belum dipakai di post #1–#3: jarak antara *repo yang bisa
> di-clone* dan *link yang bisa langsung dicoba*. Cocok diposting saat demo
> sudah live di HuggingFace Spaces. Yang dijual di sini **keputusan
> engineering**, bukan angka mAP — jangan ulang 65,5% sebagai headline.

## Versi G — Storytelling (± 340 kata, post utama)

🔗 **"Boleh lihat hasilnya?" — dan saya cuma bisa mengirim link GitHub.**

Itu yang terjadi minggu lalu. Proyek PPE Detection saya sudah lengkap:
YOLOv8 dilatih sendiri, OpenVINO INT8, REST API, UI Streamlit, 51 unit test.
Tapi untuk mencobanya, orang harus clone repo, bikin virtualenv, install 2 GB
dependency, dan menyiapkan weights. Praktis: tidak ada yang mencoba.

Jadi minggu ini saya menambahkan satu hal — frontend web satu file yang
dilayani langsung oleh FastAPI. Tanpa build step, tanpa npm. Buka URL, upload
gambar atau nyalakan webcam, selesai.

Yang ternyata menarik justru keputusan-keputusan kecilnya:

📦 **Berhenti mengirim gambar hasil.** Awalnya `/predict` selalu membalas
gambar teranotasi sebagai base64 PNG. Untuk `curl` itu enak. Untuk webcam
30 fps itu bencana: **819.914 byte per frame**. Saya tambahkan
`?annotate=false` yang hanya mengembalikan koordinat — **1.009 byte, 99,9%
lebih kecil** — lalu browser menggambar sendiri bounding box-nya di `<canvas>`.
Encode PNG di server ternyata lebih mahal daripada inference-nya.

🚦 **Satu request in-flight.** Kalau timer frame berbunyi sementara request
sebelumnya belum dijawab, frame itu saya buang. Tanpa itu, server yang lebih
lambat dari laju frame akan menumpuk antrean dan overlay makin tertinggal
dari gambar yang tampil — deteksi yang "benar" tapi milik tiga detik lalu.

🐛 **Bug yang hampir lolos.** Awalnya semua frame saya re-encode JPEG di
browser sebelum dikirim. Hemat, tapi mode gambar jadi memberi **6 objek**
sementara CLI memberi **7** untuk file yang sama — satu deteksi lemah
(`hand_glove`, confidence 0,38) hilang ditelan kompresi. Selisih satu objek,
tapi persis jenis hal yang bikin orang tidak percaya sistemnya saat demo.
Sekarang mode gambar mengirim file aslinya apa adanya.

Yang paling saya syukuri: bug itu ketahuan karena smoke test frontend-nya
memeriksa **piksel canvas** — berapa piksel merah, berapa hijau — bukan
sekadar "tidak ada error di console".

Repo + demo di komentar 👇

#ComputerVision #YOLOv8 #FastAPI #OpenVINO #EdgeAI #WebDev #JavaScript
#Docker #HuggingFace #PPEDetection #K3 #HSE #AISuperClass #RubyThalibAI

---

## Versi H — Ringkas (± 140 kata)

🔗 Proyek PPE Detection saya sudah lengkap — YOLOv8, OpenVINO INT8, REST API,
51 unit test. Tapi untuk mencobanya orang harus clone repo dan install 2 GB
dependency. Praktis tidak ada yang mencoba.

Minggu ini saya tambahkan frontend satu file HTML yang dilayani langsung oleh
FastAPI. Buka URL, upload gambar atau nyalakan webcam, selesai.

Tiga keputusan yang bikin bedanya:

📦 `?annotate=false` — endpoint balas koordinat saja, browser yang menggambar
box. Per frame: **819.914 → 1.009 byte (99,9% lebih kecil)**.
🚦 Satu request in-flight; frame yang datang saat server masih sibuk dibuang,
supaya overlay tidak tertinggal dari gambar.
🐛 Berhenti re-encode JPEG di browser — sempat menghilangkan 1 deteksi lemah
sehingga UI dan CLI memberi jawaban berbeda untuk file yang sama.

Ketahuan karena smoke test-nya memeriksa piksel canvas, bukan cuma error console.

Repo + demo di komentar 👇

#ComputerVision #YOLOv8 #FastAPI #EdgeAI #JavaScript #PPEDetection #AISuperClass

---

## Saran lampiran (post #4)

| Slide / media | Sumber | Kenapa |
|---|---|---|
| 1 | `docs/screenshots/web_ui.png` | Hasil akhir langsung terlihat: box merah/hijau + panel compliance |
| 2 | Screen recording 10–15 detik mode webcam | Bukti realtime — paling kuat, dan paling jarang dipunya orang |
| 3 | Screenshot Swagger `/docs` dengan parameter `annotate` | Menunjukkan API-nya beneran dirancang, bukan satu endpoint asal jadi |
| 4 | Potongan output `run_selftest.py` (baris "piksel : ... px merah / px hijau") | Bukti testing sampai level piksel — pembeda paling kuat buat recruiter teknis |

Catatan:

- **Angka adalah jangkarnya.** 819.914 → 1.009 byte lebih mudah diingat
  daripada "dioptimasi". Tulis angka penuhnya, jangan dibulatkan jadi "±800 KB".
- Bagian bug (`6 vs 7 objek`) adalah bagian paling manusiawi dari post ini —
  jangan dihapus untuk terdengar lebih mulus. Justru itu yang menunjukkan
  kamu menguji karyamu sendiri, bukan cuma membangunnya.
- Kalau demo HuggingFace Spaces belum live saat posting, ganti CTA jadi
  "repo di komentar" saja dan simpan kata "demo" untuk post berikutnya.
