# Konsep Deep Learning untuk Klasifikasi (Multiclass & Multi-label)

Dalam *deep learning*, klasifikasi adalah tugas fundamental untuk mengkategorikan data input. Kita akan membahas dua skenario utama berdasarkan sifat kelasnya: **Mutually Exclusive Classes (Multiclass Classification)** dan **Non-Exclusive Classes (Multi-label Classification)**.

---

## 1. Multiclass Classification (Mutually Exclusive Classes)

Gambar pertama mengilustrasikan arsitektur dasar untuk *multiclass classification*. Dalam skenario ini, setiap input (misalnya, piksel gambar atau deskripsi warna) hanya boleh termasuk ke dalam satu dan hanya satu kelas saja. Contohnya, sebuah piksel gambar tidak bisa sekaligus berwarna "Merah" dan "Biru" pada saat yang bersamaan jika kita mengklasifikasikan warna dasarnya.

### Arsitektur Jaringan Saraf Tiruan:

* **Input Layer (Lingkaran Ungu):** Menerima fitur-fitur dari data input (misalnya, nilai RGB dari sebuah piksel, atau fitur-fitur lain yang mendeskripsikan warna).
* **Hidden Layers (Lingkaran Biru):** Satu atau lebih lapisan tersembunyi yang mempelajari representasi fitur yang semakin kompleks dari data warna.
* **Output Layer (Lingkaran Hijau):** Jumlah neuron di lapisan ini sama dengan jumlah total kelas warna (N) yang ingin diprediksi (misalnya, Merah, Hijau, Biru, Kuning, dll.).

### Fungsi Aktivasi Softmax untuk Output Layer:

Untuk masalah *multiclass classification* yang *mutually exclusive* (misalnya, mengidentifikasi satu warna dominan dari serangkaian pilihan), fungsi aktivasi **Softmax** adalah pilihan yang paling umum dan tepat pada lapisan output. Softmax mengubah *raw scores* (sering disebut *logits*) dari neuron output menjadi distribusi probabilitas, di mana jumlah semua probabilitas untuk setiap kelas warna selalu 1.

Secara matematis, untuk sebuah neuron output $i$ dengan *logit* $z_i$, probabilitas kelas $p_i$ dihitung sebagai:

$$p_i = \frac{e^{z_i}}{\sum_{j=1}^{N} e^{z_j}}$$

* $e^{z_i}$ adalah eksponensial dari *logit* untuk kelas warna $i$.
* $\sum_{j=1}^{N} e^{z_j}$ adalah jumlah eksponensial dari semua *logits* untuk semua $N$ kelas warna.

### Contoh Output Softmax (dari Gambar Kedua):

Jika kita memiliki 3 kelas warna: `[Red, Green, Blue]`, dan Softmax menghasilkan output: `[0.1, 0.6, 0.3]`, ini berarti:

* Probabilitas input adalah `Red`: 10%
* Probabilitas input adalah `Green`: 60%
* Probabilitas input adalah `Blue`: 30%

Dalam kasus ini, model akan memprediksi input tersebut memiliki warna `Green` karena memiliki probabilitas tertinggi (0.6). Penting untuk dicatat bahwa $0.1 + 0.6 + 0.3 = 1.0$, menunjukkan bahwa semua probabilitas telah dinormalisasi dan saling melengkapi.

---

## 2. Multi-label Classification (Non-Exclusive Classes)

Meskipun gambar yang Anda berikan berfokus pada *mutually exclusive classes* (yang menggunakan Softmax), penting untuk memahami bahwa ada skenario di mana kelas-kelas warna tidak saling eksklusif. Ini disebut *Multi-label Classification*.

Dalam *multi-label classification*, sebuah input tunggal (misalnya, deskripsi campuran warna dalam sebuah objek) dapat termasuk ke dalam beberapa kelas warna secara bersamaan. Contohnya adalah mengklasifikasikan corak atau pola yang memiliki campuran beberapa warna, misalnya `[Merah, Kuning, Ungu]` pada sebuah kain batik.

### Perbedaan Utama dalam Arsitektur & Fungsi Aktivasi:

* **Output Layer:** Jumlah neuron di lapisan output tetap sama dengan jumlah total kelas warna yang mungkin.
* **Fungsi Aktivasi:** Berbeda dengan Softmax, untuk setiap neuron di lapisan output kita akan menggunakan fungsi aktivasi **Sigmoid**.
* **Sigmoid:** Fungsi Sigmoid menghasilkan output antara 0 dan 1 untuk setiap neuron secara independen.

Secara matematis, untuk sebuah neuron output $i$ dengan *logit* $z_i$, probabilitas kelas $p_i$ dihitung sebagai:

$$p_i = \frac{1}{1 + e^{-z_i}}$$

Setiap $p_i$ mewakili probabilitas bahwa input tersebut mengandung warna $i$, terlepas dari warna lainnya. Jika output untuk sebuah input yang menggambarkan campuran warna adalah `[Merah: 0.9, Kuning: 0.2, Ungu: 0.8]`, itu berarti objek ini sangat mungkin memiliki Merah dan Ungu, tetapi tidak terlalu Kuning.

---

### Tabel Perbandingan Singkat:

| Fitur | Multiclass Classification (Saling Eksklusif) | Multi-label Classification (Tidak Saling Eksklusif) |
| :-------------------------- | :------------------------------------------- | :-------------------------------------------------- |
| **Definisi** | Satu input = satu kelas dominan | Satu input = bisa beberapa kelas |
| **Fungsi Aktivasi Output** | Softmax | Sigmoid (diterapkan per neuron) |
| **Total Probabilitas** | Jumlah probabilitas seluruh kelas = 1 | Probabilitas setiap kelas dihitung independen |
| **Fungsi Kerugian Umum** | Cross-Entropy Kategorikal | Cross-Entropy Biner (diterapkan per kelas) |

---

Memahami perbedaan mendasar ini akan sangat membantu Anda dalam merancang, mengimplementasikan, dan melatih model *deep learning* yang tepat sesuai dengan sifat spesifik dari tugas klasifikasi warna Anda.
