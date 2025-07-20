# Memahami Activation Function dalam Deep Learning

## Konsep Activation Function

Dalam dunia *artificial neural network*, **Activation Function (Fungsi Aktivasi)** adalah komponen kunci pada setiap **node** (atau neuron). Fungsinya adalah untuk menghitung output dari node berdasarkan input yang diterima dan bobot yang terkait.

Fungsi aktivasi **non-linear** sangat penting untuk memecahkan masalah kompleks yang tidak bisa diselesaikan dengan transformasi linear sederhana. Tanpa non-linearitas ini, sebuah *neural network* akan setara dengan model regresi linear, tidak peduli seberapa banyak lapisannya.

Konsep ini didukung oleh **Universal Approximation Theorem**, yang menyatakan bahwa jaringan saraf dua-lapisan dengan fungsi aktivasi non-linear dapat mengaproksimasi fungsi kontinu apa pun. Inilah yang membuat *deep learning* begitu kuat.

### Properti Penting Activation Function:

* **Non-linear:** Memungkinkan jaringan mempelajari hubungan dan pola data yang kompleks.
* **Rentang (Range):** Output terbatas (misalnya, 0-1 atau -1-1) umumnya menghasilkan pelatihan berbasis gradien yang lebih stabil. Rentang tak terbatas bisa lebih efisien tetapi mungkin memerlukan *learning rate* yang lebih kecil.
* **Dapat Didiferensiasi secara Kontinu (Continuously Differentiable):** Penting untuk metode optimasi berbasis gradien (**backpropagation**), memungkinkan penyesuaian bobot yang efisien.

---

## Jenis-Jenis Activation Function

Fungsi aktivasi dapat dikategorikan menjadi beberapa jenis utama:

### 1. Ridge Functions

Kategori ini mencakup fungsi yang beroperasi pada kombinasi linear dari variabel input.

* **Linear Activation**
    Fungsi paling sederhana yang tidak memperkenalkan non-linearitas.
    * **Rumus:** $\phi(\mathbf{v}) = a + \mathbf{v}'\mathbf{b}$
        (Di mana $\mathbf{v}$ adalah vektor input, $\mathbf{b}$ adalah vektor bobot, dan $a$ adalah bias.)
    * **Penggunaan:** Umumnya pada **lapisan output untuk masalah regresi**. Jarang di lapisan tersembunyi karena tidak dapat mempelajari pola non-linear.

* **ReLU (Rectified Linear Unit) Activation**
    Salah satu fungsi aktivasi paling populer di *deep learning* modern. Mengembalikan input jika positif, dan nol jika negatif.
    * **Rumus:** $\phi(\mathbf{v}) = \max(0, a + \mathbf{v}'\mathbf{b})$
    * **Penggunaan:** Sangat umum di **lapisan tersembunyi** *deep neural networks* karena efisiensi komputasi dan kemampuannya mengatasi masalah *vanishing gradient*.

* **Heaviside Activation (Binary Step Function)**
    Fungsi *threshold* biner. Outputnya 1 jika input melebihi *threshold*, dan 0 jika tidak.
    * **Rumus:** $\phi(\mathbf{v}) = 1_{a+\mathbf{v}'\mathbf{b}>0}$
        (Di mana $1_{kondisi}$ adalah fungsi indikator yang bernilai 1 jika kondisi benar, dan 0 jika salah.)
    * **Penggunaan:** Lebih relevan secara historis dalam model *perceptron* awal. Jarang digunakan modern karena tidak dapat didiferensiasi.

* **Logistic (Sigmoid) Activation**
    Memampatkan input ke dalam rentang 0 hingga 1, sering diinterpretasikan sebagai probabilitas.
    * **Rumus:** $\phi(\mathbf{v}) = (1+\exp(-a-\mathbf{v}'\mathbf{b}))^{-1}$ atau $f(z) = \frac{1}{1 + e^{-z}}$
    * **Penggunaan:** Cocok untuk **lapisan output dalam masalah klasifikasi biner**. Rentan terhadap masalah *vanishing gradient* untuk input ekstrem.

### 2. Radial Activation Functions (RBFs)

Digunakan dalam jaringan RBF (Radial Basis Function Networks), fungsi-fungsi ini mengukur kedekatan input dengan "pusat" tertentu.

* **Gaussian**
    * **Rumus:** $\phi(\mathbf{v}) = \exp\left(-\frac{\|\mathbf{v}-\mathbf{c}\|^{2}}{2\sigma^{2}}\right)$
        (Di mana $\mathbf{c}$ adalah pusat, $\sigma$ adalah parameter lebar, dan $\|\cdot\|$ adalah norma Euclidean.)
* **Multiquadratics**
    * **Rumus:** $\phi(\mathbf{v}) = \sqrt{\|\mathbf{v}-\mathbf{c}\|^{2}+a^{2}}$
        (Di mana $a$ adalah parameter konstan.)
* **Inverse Multiquadratics**
    * **Rumus:** $\phi(\mathbf{v}) = \left(\|\mathbf{v}-\mathbf{c}\|^{2}+a^{2}\right)^{-\frac{1}{2}}$
* **Penggunaan (untuk semua RBFs):** Digunakan dalam arsitektur jaringan RBF yang berbeda dari *multilayer perceptron* konvensional.

### 3. Folding Activation Functions

Umumnya digunakan dalam *pooling layers* pada *convolutional neural networks* dan *output layers* pada jaringan klasifikasi *multiclass*.

* **Softmax Activation**
    Mengubah vektor nilai numerik menjadi distribusi probabilitas, di mana setiap elemen berada di antara 0 dan 1, dan jumlah semua elemen adalah 1.
    * **Rumus:** $P(y=j | \mathbf{z}) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}}$
        (Di mana $z_j$ adalah input untuk kelas $j$, dan $K$ adalah jumlah total kelas.)
    * **Penggunaan:** Hampir secara eksklusif pada **lapisan output dalam masalah klasifikasi multi-kelas**.

---

### Contoh Modern dan Lainnya:

* **GELU (Gaussian Error Linear Unit)**
    Alternatif ReLU yang lebih halus, menunjukkan kinerja baik dalam model terbaru.
    * **Rumus:** $GELU(x) = x \cdot \Phi(x)$
        (Di mana $\Phi(x)$ adalah Fungsi Distribusi Kumulatif Normal Standar.)
    * **Penggunaan:** Populer dalam model *transformer* seperti **BERT**, seringkali memberikan hasil yang lebih baik daripada ReLU.

* **Fungsi Periodik (Contoh: Sinusoid)**
    Meskipun kurang umum, dapat digunakan dalam skenario khusus (misalnya, data temporal dengan pola berulang).
    * **Rumus Contoh (Sinusoid):** $f(x) = \sin(x)$

* **Quadratic Activation**
    Mengembalikan kuadrat dari input.
    * **Rumus:** $f(x) = x^2$
    * **Penggunaan:** Jarang dalam arsitektur umum karena potensi masalah *exploding/vanishing gradient*.

---

**Catatan:** Fungsi aktivasi **non-saturating** (seperti ReLU dan variannya) seringkali lebih disukai daripada yang saturasi (seperti Sigmoid atau Tanh) karena cenderung tidak mengalami masalah **vanishing gradient**, yang dapat menghambat pelatihan jaringan yang sangat dalam.
