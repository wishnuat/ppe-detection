# Draft Caption LinkedIn — PPE Detection Project

> Bahasa Indonesia · Siap posting · Ganti `@INSTRUKTUR` dan `@AI_SUPER_CLASS`
> dengan handle sesuai instruksi tugas kelas.

---

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

## Catatan pemakaian

- **Tag instruktur:** ganti `@INSTRUKTUR` dan `@AI_SUPER_CLASS` dengan
  handle LinkedIn resmi yang diminta pihak kelas.
- **Repo/demo di komentar:** siapkan link GitHub repo, screenshot, atau
  short-clip demo (max 30 detik) dan taruh di komentar pertama supaya
  post utama tetap ringkas.
- **Waktu posting:** LinkedIn engagement Indonesia biasanya bagus di jam
  08.00–10.00 atau 19.00–21.00 WIB.
- **Media:** post carousel 3–5 slide (arsitektur diagram, screenshot UI,
  screenshot Swagger, GIF deteksi) menaikkan reach signifikan.
