# CNN Prediksi Jenis Ikan 🐟

Proyek ini adalah implementasi Convolutional Neural Network (CNN) menggunakan TensorFlow dan Keras untuk memprediksi jenis ikan berdasarkan gambar. 
Model ini dilatih untuk mengenali 3 jenis ikan:
1. **Kembung**
2. **Gerabah**
3. **Tongkol**

---

## 📁 Struktur Dataset
Sebelum menjalankan script, pastikan Anda telah menyusun gambar-gambar ikan ke dalam folder sesuai struktur berikut:

```
dataset_ikan/
├── train/              <-- (Untuk melatih AI)
│   ├── gerabah/
│   ├── kembung/
│   └── tongkol/
└── validation/         <-- (Untuk menguji/mengevaluasi AI)
    ├── gerabah/
    ├── kembung/
    └── tongkol/
```
**Tips:** Masukkan sekitar 80% gambar ke folder `train` dan 20% gambar sisanya ke folder `validation`.

---

## ⚙️ Persiapan & Instalasi

Pastikan Anda memiliki **Python 3.9** (atau versi yang kompatibel dengan TensorFlow).

**1. Buat Virtual Environment (Sangat Direkomendasikan)**
```bash
# Untuk pengguna Mac / Linux:
/usr/bin/python3 -m venv .venv
source .venv/bin/activate
```

**2. Install Library yang Dibutuhkan**
```bash
pip install tensorflow matplotlib scikit-learn seaborn numpy
```

---

## 🚀 Cara Menjalankan Program

### 1. Melatih Model (Training)
Setelah dataset disiapkan, jalankan perintah berikut untuk mulai melatih model:
```bash
python train_cnn.py
```
**Apa yang akan dihasilkan?**
- `model_prediksi_ikan.h5`: File model utama yang sudah "pintar".
- `learning_curves.png`: Grafik performa dari akurasi dan _loss_ selama pelatihan.
- *(Jika ada folder validation)* `confusion_matrix.png`: Matriks visualisasi tebakan benar vs salah.
- *(Jika ada folder validation)* `evaluation_metrics.txt`: Rincian akurasi per kelas.

### 2. Menguji Model (Testing / Prediksi)
Jika Anda ingin memprediksi sebuah gambar secara acak untuk mengetes AI-nya, gunakan perintah berikut:
```bash
python test_cnn.py <lokasi/file/gambar.jpg>
```
**Contoh:**
```bash
python test_cnn.py dataset_real/validation/tongkol/ikan1.jpg
```
Nantinya program akan mengeluarkan hasil persentase keyakinan tebakan seperti: "Jenis Ikan: TONGKOL, Tingkat Yakin: 98.50%".

---

## 🔬 Modul Riset (Standar Jurnal SINTA 2)

### A. Explainable AI (Grad-CAM Heatmap)
Untuk melihat area mana pada ikan yang difokuskan oleh AI dalam mengambil keputusan:
```bash
python grad_cam.py dataset_real/validation/kembung_segar/gambar_01.jpg
```
*Script ini akan menghasilkan gambar `gradcam_plot.png`.*

### B. Studi Komparasi (Ablation Study)
Untuk membuktikan bahwa arsitektur MobileNetV2 lebih baik dibandingkan CNN standar buatan sendiri:
```bash
python ablation_study.py
```
*Script ini melatih 2 model secara paralel dan menghasilkan grafik perbandingan di `ablation_study_results.png`.*
