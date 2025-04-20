- # Proyek Klasifikasi Gambar Fashion MNIST menggunakan CNN dan TensorFlow

## Deskripsi
Proyek ini adalah implementasi sistem klasifikasi gambar menggunakan dataset Fashion MNIST. Dataset ini terdiri dari 10 kategori pakaian, seperti T-shirt/top, Trouser, Pullover, Dress, dan lainnya. Model yang digunakan adalah Convolutional Neural Network (CNN) dengan TensorFlow dan Keras. Proyek ini juga mendemonstrasikan konversi model ke beberapa format untuk keperluan deployment: SavedModel, TensorFlow.js, dan TensorFlow Lite.

## Fitur Utama
1. **Pemrosesan Data**:
   - Normalisasi dataset Fashion MNIST agar nilai piksel berada dalam rentang [0, 1].
   - Transformasi data untuk kompatibilitas dengan CNN (dimensi 28x28x1).

2. **Arsitektur CNN**:
   - 2 blok convolutional dan pooling.
   - Lapisan fully connected dengan aktivasi ReLU.
   - Lapisan output dengan 10 kelas menggunakan aktivasi softmax.

3. **Evaluasi Model**:
   - Evaluasi akurasi pada data training, validasi, dan testing.
   - Visualisasi kurva akurasi dan loss.

4. **Simpan Model**:
   - Simpan model dalam format SavedModel, TensorFlow.js, dan TensorFlow Lite.

5. **Deployment**:
   - Model siap digunakan dalam aplikasi web atau perangkat mobile.

## Cara Menjalankan
1. **Persiapan Lingkungan**:
   - Pastikan Python 3.x sudah terinstal.
   - Install dependensi menggunakan `requirements.txt`.

   ```bash
   pip install -r requirements.txt

