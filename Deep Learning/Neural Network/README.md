# Konsep Neural Network

## 1. Arsitektur Jaringan Saraf (Neural Network) dan Parameter

Sebuah jaringan saraf tiruan (Artificial Neural Network/ANN) terdiri dari lapisan-lapisan neuron yang saling terhubung. Setiap koneksi antar neuron memiliki bobot (weight) dan setiap neuron memiliki bias. Proses ini memungkinkan jaringan untuk mempelajari pola dari data.

### Struktur Jaringan dan Parameter
Mari kita lihat contoh jaringan saraf kecil dengan semua parameternya diberi label:

![Small Network with Labeled Parameters](image-1.png)

Dalam ilustrasi ini:
* **$I_1, I_2, I_3$**: Adalah input ke lapisan pertama (Layer 1).
* **$w_{j,k}^l$**: Merujuk pada **bobot (weight)** yang menghubungkan neuron ke-$k$ di lapisan $l-1$ ke neuron ke-$j$ di lapisan $l$. Misalnya, $w_{1,1}^2$ adalah bobot dari input $I_1$ ke neuron pertama di Layer 2.
* **$b_j^l$**: Merujuk pada **bias** untuk neuron ke-$j$ di lapisan $l$. Misalnya, $b_1^2$ adalah bias untuk neuron pertama di Layer 2.
* **$a_j^l$**: Merujuk pada **aktivasi** (output) dari neuron ke-$j$ di lapisan $l$ setelah melalui fungsi aktivasi.
* **$O_1, O_2$**: Adalah output akhir dari jaringan (Layer 4).

Setiap neuron di lapisan tersembunyi menghitung outputnya (aktivasi) dengan menjumlahkan hasil perkalian input dari lapisan sebelumnya dengan bobotnya masing-masing, ditambah bias, kemudian menerapkan fungsi aktivasi (misalnya, fungsi sigmoid seperti yang digambarkan oleh ikon kurva "S" kecil).

**Ringkasan Parameter:**
* **W (Weights / Bobot):** Matriks bobot yang menentukan kekuatan koneksi antar neuron.
* **B (Biases / Bias):** Vektor bias yang ditambahkan ke input yang dibobotkan, memungkinkan neuron untuk mengaktifkan dirinya bahkan ketika semua inputnya nol.

