## Konsep Perceptron
![Perceptron Diagram](Perceptron.PNG)

Perceptron bekerja dengan mengambil sejumlah input, mengalikan setiap input dengan bobotnya masing-masing, menjumlahkan semua hasil perkalian tersebut bersama dengan nilai bias, dan kemudian melewati total ini melalui fungsi aktivasi untuk menghasilkan output akhir.

### Variabel Utama

Dalam konteks Perceptron, terdapat beberapa variabel kunci:

* **Input ($x_1, x_2, \ldots, x_n$):** Ini adalah fitur atau data masukan yang diberikan kepada model. Dalam konetks ini, kita mungkin akan melihat contoh dengan dua input, $x_1$ dan $x_2$. 

* **Bobot (Weights) ($w_1, w_2, \ldots, w_n$):** Setiap input ($x_i$) memiliki bobot ($w_i$) yang sesuai. Bobot ini merepresentasikan **pentingnya atau kekuatan pengaruh** dari input tersebut terhadap keputusan akhir model.

* **Bias ($b$):** Bias adalah nilai konstan yang ditambahkan ke jumlah tertimbang dari input. Ini berfungsi sebagai **ambang batas (threshold) internal** yang memungkinkan model untuk menggeser garis pemisah keputusannya.

* **Jumlah Tertimbang ($X$):** Ini adalah hasil penjumlahan semua perkalian input dengan bobotnya, ditambah bias.
    $$X = (x_1 \cdot w_1) + (x_2 \cdot w_2) + \ldots + (x_n \cdot w_n) + b$$

* **Fungsi Aktivasi $f$($X$):** Sebuah fungsi non-linear yang diterapkan pada $X$ untuk menghasilkan output akhir. Untuk perceptron klasik, seringkali digunakan fungsi langkah (step function).
    $$y = f(X)$$

* **Output ($y$):** Ini adalah prediksi yang dihasilkan oleh Perceptron, biasanya berupa nilai biner (0 atau 1) untuk klasifikasi.



