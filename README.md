
# KosIn: Jabodetabek Boarding Room Price Predictor

> Mengestimasi harga sewa kamar kos ideal di wilayah DKI Jakarta menggunakan algoritma Machine Learning berbasis spesifikasi properti, kebijakan sewa, dan kelengkapan fasilitas.

**Task:** Regression

**Framework:** Pandas, NumPy, Scikit-Learn, Streamlit

**Default Branch:** feature-ui 
---

## Results

[cite_start]Selama tahap pengembangan model, kami melatih algoritma *baseline* sebagai *benchmark selection* [cite_start]Berikut adalah perbandingan metrik performa akhir antara model *baseline* dengan model terpilih

| Model | $R^2$ Score | MAE | MSE | RMSE |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | 0.7085 | Rp 294.007 | Rp 175.013.337.810 | Rp 418.346 |
| **Random Forest Regressor (Chosen)** | **0.8105** | **Rp 240.958** | **Rp 148.889.345.611** | **Rp 385.862** |

*Catatan: Nilai $R^2$ Score dihitung pada skala logaritma (Log Transformation Target) demi menstabilkan varians, sedangkan MAE dan RMSE telah dikembalikan menggunakan fungsi eksponensial ke dalam satuan Rupiah asli agar mudah diinterpretasikan oleh pengguna*

---

## Setup

**Requirements:** Python `3.10+`

```bash
git clone [(https://github.com/aqilanailalhusna/prediksi_harga_kos)]
cd kosin-predict-ml
pip install -r requirements.txt

```

---

## Data

Dataset yang digunakan dalam proyek ini bersifat *open-source* yang diperoleh melalui platform Kaggle: [Link Dataset](https://www.kaggle.com/datasets/dendykurniariagman/mamikos-jabodetabek-boarding-room-listings).

### EDA & Preprocessing Insight

* **Filter Wilayah Geografis:** Data mentah mencakup area Jabodetabek , namun direduksi khusus untuk wilayah DKI Jakarta (menjadi 1.233 baris) karena dinamika harga di pusat kota jauh lebih fluktuatif.


* **Disparitas Wilayah:** Berdasarkan analisis geografi, Jakarta Selatan menempati rata-rata harga kos tertinggi (Rp 2.337.112), sedangkan Jakarta Timur menjadi opsi yang paling terjangkau (Rp 1.549.845).


* **Korelasi Fasilitas Pokok:** Berdasarkan visualisasi matriks korelasi (*heatmap*), ketersediaan fasilitas premium seperti `fac_air_panas` (water heater) dan `fac_tv` memiliki hubungan positif paling kuat terhadap kenaikan harga sewa.


* **Log Transformation Target:** Kolom harga asli (`price`) dibersihkan dari komponen string teks simbol mata uang, lalu ditransformasikan ke dalam nilai logaritma untuk mengatasi masalah sebaran data pencilan ekstrem (*right-skewed*).



### Format Atribut

Data akhir memiliki total 1.233 baris dengan kombinasi fitur utama sebagai berikut:

* **Geografis & Deskriptif:** `region` (Kota administrasi), `room_area` (Luas kamar dalam $m^2$), `tipe_kos` (Campur, Putra, Putri).


* **Fasilitas Bundling (Hasil Preprocessing):**
* `is_kamar_mandi_dalam` & `is_kloset_duduk` (Indikator komponen kamar mandi modern).
* `is_full_furnished` (Otomatis bernilai 1 jika memiliki komponen Kasur, Lemari Baju, dan Meja sekaligus).
* `fac_keamanan` (Gabungan biner dari fitur CCTV, Penjaga Kos, dan Pengurus Kos).
* `fac_ruang_bersama` (Gabungan komponen Ruang Tamu, Ruang Makan, Ruang Santai, Balkon, dan Rooftop).
* `fac_layanan_kebersihan` & `fac_area_jemur` (Komponen efisiensi kebersihan cuci-jemur).


* **Fasilitas Populer Individu (> 50 kos):** `fac_ac`, `fac_wifi`, `fac_tv`, `fac_air_panas`, `fac_kulkas`, `is_electricity` (Status biaya listrik biner), dan `is_discount` (Logika matematis potongan harga).



### Splits

Pembagian data menggunakan strategi *Train-Test Split* acak dengan parameter proporsi seimbang:

* **Data Train (80%):** 986 baris 


* **Data Test (20%):** 247 baris 



---

## Usage

Untuk menjalankan aplikasi antarmuka KosIn berbasis Streamlit local server:

```bash
streamlit run app.py

```

---

## Deployment

Aplikasi dapat diakses melalui link : https://kosin-pred.streamlit.app/

---

## Model & Architecture

Algoritma **Random Forest Regressor** dipilih karena terbukti jauh lebih akurat dan tangguh dalam menangkap pola hubungan harga properti sewa yang bersifat non-linear serta dipengaruhi oleh kombinasi variabel fasilitas yang kompleks.

Aplikasi ini didesain menggunakan **Streamlit Architecture**, di mana komponen *FrontEnd* (antarmuka) dan *BackEnd* (logika prediksi ML) berjalan secara monolitik di dalam satu lingkungan Python yang sama. Berkas beban model yang sudah dilatih disimpan langsung dalam direktori proyek berformat serialisasi `.pkl`.

---

## User Testing Design & Results

### Design

Pengujian dilakukan terhadap 5 responden pengguna mahasiswa menggunakan skenario pencarian langsung (*Usage Scenario*):

* **Input Parameters:** Pengguna memasukkan kriteria berupa lokasi lewat *dropdown*, luas kamar lewat komponen *slider*, tipe kos menggunakan *radio button*, dan mencentang fasilitas lewat *checkbox*.


* **Output Screen:** Aplikasi mengembalikan nominal estimasi harga ideal per bulan dan memunculkan komponen daftar Top 5 Rekomendasi Kos yang paling mendekati kriteria kueri input.



### Results (Feedback)

* **Usability (100% Puas):** Seluruh responden memberikan nilai kepuasan tertinggi (skala 4/4) untuk aspek kerapian tata letak menu dan kemudahan alur pemakaian aplikasi Streamlit.


* **Usefulness:** Informasi tebakan harga dinilai sangat masuk akal dan relevan dengan realita pasar kos Jakarta karena tingkat kesalahan rata-rata model (MAE) berada di bawah batas toleransi anggaran psikologis penyewa (< Rp 250.000).


* **Latency:** Proses kalkulasi komputasi dari penekanan tombol prediksi hingga memunculkan hasil output dirasakan berjalan sangat cepat (di bawah 1 detik).



---

## Limitations

### Ketiadaan Faktor Eksternal Penentu Properti

Model estimasi saat ini murni hanya membaca data spesifikasi fisik komponen internal bangunan kos (fasilitas, wilayah administrasi, ukuran kamar). Model belum mampu mengintegrasikan variabel eksternal krusial di dunia nyata seperti jarak radius menuju stasiun KRL/halte TransJakarta terdekat, kedekatan dengan area kampus, atau status kerawanan wilayah terhadap banjir.

### Gap Nilai RMSE (Efek Outlier Properti)

Meskipun nilai kesalahan rata-rata linear (MAE) sudah sangat rendah , nilai RMSE model masih menyentuh angka Rp 388.647. Hal ini mengindikasikan adanya efek penalti dari sisa data pencilan (*outliers*) properti sewa tipe mewah yang harganya melonjak drastis akibat faktor eksklusivitas kualitatif lingkungan yang fiturnya belum terekam dalam bentuk teks terstruktur di dataset.

### Sistem Bersifat Read-Only

Berdasarkan keluhan umpan balik pengguna pada pengujian kualitatif, aplikasi KosIn saat ini masih terbatas sebagai instrumen edukasi transparansi harga anggaran sewa (*read-only*). Sistem belum menyediakan integrasi database atau API komunikasi dua arah yang dapat menghubungkan penyewa secara instan dengan kontak personal pemilik kos asli.
