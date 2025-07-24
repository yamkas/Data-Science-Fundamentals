# Konsep Backpropagation

Backpropagation adalah algoritma fundamental dalam deep learning yang digunakan untuk melatih jaringan saraf dengan menyesuaikan bobot dan bias untuk meminimalkan fungsi biaya. Ini melibatkan perambatan gradien kesalahan mundur melalui jaringan.

## 1. Tujuan Backpropagation

Tujuan utama backpropagation adalah untuk mengetahui bagaimana perubahan bobot jaringan memengaruhi fungsi biaya, memungkinkan jaringan untuk memperbarui bobot ini guna meminimalkan fungsi biaya.

## 2. Contoh Jaringan Sederhana

Pertimbangkan jaringan yang sangat sederhana di mana setiap lapisan hanya memiliki satu neuron. Setiap input dalam jaringan ini menerima bobot dan bias. Fungsi biaya untuk jaringan seperti itu dengan bobot $w1, w2, w3$ dan bias $b1, b2, b3$ dapat direpresentasikan sebagai $C(w1, b1, w2, b2, w3, b3)$.

## 3. Perambatan Maju (Tinjauan)

Sebelum membahas backpropagation, penting untuk memahami proses perambatan maju. Dalam jaringan saraf dengan $L$ lapisan, notasi untuk lapisan dapat berupa $L-n, L-2, L-1, L$.

Untuk setiap lapisan $L$, input berbobot $z^L$ dan aktivasi $a^L$ didefinisikan sebagai:

* $z^L = w^L a^{L-1} + b^L$
* $a^L = \sigma(z^L)$

Fungsi biaya, $C_0$, untuk lapisan terakhir dapat dinyatakan sebagai:

* $C_0(...) = (a^L - y)^2$

## 4. Sensitivitas Fungsi Biaya terhadap Bobot dan Bias

Untuk meminimalkan fungsi biaya, kita perlu memahami seberapa sensitif fungsi biaya terhadap perubahan bobot ($w$) dan bias ($b$). Ini melibatkan perhitungan turunan parsial menggunakan aturan rantai.

Untuk lapisan terakhir $L$, turunan parsial dari fungsi biaya $C_0$ terhadap bobot $w^L$ diberikan oleh:

$$\frac{\partial C_0}{\partial w^L} = \frac{\partial z^L}{\partial w^L} \frac{\partial a^L}{\partial z^L} \frac{\partial C_0}{\partial a^L}$$

Demikian pula, untuk bias $b^L$:

$$\frac{\partial C_0}{\partial b^L} = \frac{\partial z^L}{\partial b^L} \frac{\partial a^L}{\partial z^L} \frac{\partial C_0}{\partial a^L}$$

Ide utamanya adalah menggunakan informasi gradien ini untuk kembali melalui jaringan dan menyesuaikan bobot dan bias, meminimalkan output vektor kesalahan pada lapisan output terakhir.

## 5. Melatih Jaringan Saraf dengan Langkah-Langkah Backpropagation

Backpropagation melibatkan beberapa langkah kunci untuk melatih jaringan saraf, dan ini dapat diperluas menggunakan notasi kalkulus untuk jaringan dengan beberapa neuron per lapisan.

### Langkah 1: Mengatur Aktivasi untuk Lapisan Input

Menggunakan input $x$, atur fungsi aktivasi $a$ untuk lapisan input. Perhitungan melibatkan:

* $z = w x + b$
* $a = \sigma(z)$

Aktivasi yang dihasilkan $a$ kemudian masuk ke lapisan berikutnya (dan seterusnya).

### Langkah 2: Menghitung untuk Setiap Lapisan (Forward Pass)

Untuk setiap lapisan $L$, hitung input berbobot dan aktivasi:

* $z^L = w^L a^{L-1} + b^L$
* $a^L = \sigma(z^L)$

### Langkah 3: Menghitung Vektor Kesalahan ($\delta^L$)

Vektor kesalahan untuk lapisan output $L$ dihitung sebagai:

$$\delta^L = \nabla_a C \odot \sigma'(z^L)$$

Di sini, $\nabla_a C$ menyatakan laju perubahan fungsi biaya $C$ terhadap aktivasi output, dan diberikan oleh:

$$\nabla_a C = (a^L - y)$$

Oleh karena itu, vektor kesalahan dapat ditulis sebagai:

$$\delta^L = (a^L - y) \odot \sigma'(z^L)$$

Simbol $\odot$ menunjukkan produk Hadamard.

### Langkah 4: Backpropagate Kesalahan ($\delta^l$)

Untuk melakukan backpropagate kesalahan untuk lapisan $L-1, L-2, \ldots$, kesalahan umum untuk setiap lapisan $l$ dihitung sebagai:

$$\delta^l = (w^{l+1})^T \delta^{l+1} \odot \sigma'(z^l)$$

Dalam rumus ini:

* $(w^{l+1})^T$ adalah transpose dari matriks bobot dari lapisan $l+1$. Secara intuitif, menerapkan matriks bobot transpose ini memindahkan kesalahan mundur melalui jaringan, memberikan semacam ukuran kesalahan pada output lapisan ke-$l$.
* Produk Hadamard $\odot \sigma'(z^l)$ kemudian memindahkan kesalahan ini mundur melalui fungsi aktivasi di lapisan $l$, memberikan kita kesalahan $\delta^l$ pada input berbobot ke lapisan $l$.

## 6. Gradien Fungsi Biaya

Gradien fungsi biaya terhadap bobot dan bias untuk setiap lapisan $l$ diberikan oleh:

$$\frac{\partial C}{\partial w_{jk}^l} = a_k^{l-1} \delta_j^l$$

$$\frac{\partial C}{\partial b_j^l} = \delta_j^l$$

Gradien ini memungkinkan penyesuaian bobot dan bias untuk membantu meminimalkan fungsi biaya.

## Keterangan Simbol dalam Rumus

Berikut adalah penjelasan simbol-simbol yang digunakan dalam rumus-rumus di atas:

* $L$: Menunjukkan nomor lapisan dalam jaringan saraf (misalnya, lapisan output).
* $l$: Menunjukkan indeks lapisan umum dalam jaringan saraf.
* $n$: Menunjukkan indeks lapisan umum yang lebih awal.
* $w^L$: Matriks bobot untuk lapisan $L$.
* $b^L$: Vektor bias untuk lapisan $L$.
* $a^L$: Vektor aktivasi (output) dari lapisan $L$.
* $z^L$: Vektor input berbobot (sebelum fungsi aktivasi) dari lapisan $L$.
* $\sigma$: Fungsi aktivasi (misalnya, fungsi sigmoid atau ReLU).
* $\sigma'$: Turunan dari fungsi aktivasi.
* $C$: Fungsi biaya (cost function) yang ingin diminimalkan oleh jaringan.
* $C_0$: Fungsi biaya untuk output pada lapisan terakhir.
* $y$: Vektor target atau label yang benar.
* $\delta^L$: Vektor kesalahan untuk lapisan output $L$.
* $\delta^l$: Vektor kesalahan untuk lapisan $l$.
* $\nabla_a C$: Gradien fungsi biaya $C$ terhadap aktivasi $a$.
* $\odot$: Produk Hadamard (elemen-demi-elemen).
* $(w^{l+1})^T$: Transpose dari matriks bobot dari lapisan $l+1$.
* $\frac{\partial C}{\partial w_{jk}^l}$: Turunan parsial dari fungsi biaya $C$ terhadap bobot $w$ yang menghubungkan neuron $k$ di lapisan $l-1$ dengan neuron $j$ di lapisan $l$.
* $\frac{\partial C}{\partial b_j^l}$: Turunan parsial dari fungsi biaya $C$ terhadap bias $b$ dari neuron $j$ di lapisan $l$.
