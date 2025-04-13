# Proyek-Klasifikasi-Gambar
Deskripsi Proyek
Klasifikasi gambar adalah salah satu aplikasi populer dalam machine learning. Proyek ini berfokus pada:
Dataset CIFAR-10 yang memiliki 10 kelas gambar, seperti pesawat (airplane), mobil (car), burung (bird), dll.
Model CNN dengan lapisan Convolutional (Conv2D), MaxPooling, dan Dense untuk klasifikasi.
Konversi Model ke format SavedModel, TF-Lite, dan TFJS untuk mempermudah deployment.
Proyek ini diimplementasikan dalam Google Colab, dengan fokus pada pelatihan, validasi, dan pengujian model.

# Struktur Proyek
submission/
├───tfjs_model/
│   ├───group1-shard1of1.bin
│   └───model.json
├───tflite/
│   ├───model.tflite
│   └───label.txt
├───saved_model/
│   ├───saved_model.pb
│   └───variables/
├───notebook.ipynb
├───README.md
└───requirements.txt

# Langkah-langkah Implementasi
1. Persiapan Dataset:
   - Dataset CIFAR-10 dimuat langsung dari TensorFlow.
   - Dataset dibagi menjadi 80% untuk train set, 20% untuk validation set, dan 10.000 gambar untuk test set.
2. Preprocessing:
   - Normalisasi gambar dilakukan dengan membagi nilai piksel gambar dengan 255.
3. Model CNN:
   - Model Sequential dengan 3 lapisan Conv2D dan MaxPooling.
   - Lapisan Dense digunakan untuk klasifikasi dengan 10 output kelas.
4. Pelatihan Model:
   - Callback EarlyStopping digunakan untuk menghentikan pelatihan saat akurasi tidak meningkat.
   - Model dilatih selama 10 epoch untuk menghemat waktu, dengan hasil akurasi training set mencapai 85%+.
5. Evaluasi Model:
   - Model diuji pada test set, menghasilkan akurasi yang tinggi (>85%).
6. Visualisasi:
   - Plot akurasi dan loss disediakan untuk memahami performa pelatihan.
7. Konversi Model:
   - Model disimpan dalam format SavedModel, TF-Lite, dan TFJS.
   - Format TF-Lite cocok untuk perangkat mobile.
   - Format TFJS memungkinkan model dijalankan di browser.
8. Inferensi Model:
   - Inferensi dilakukan untuk memprediksi kelas gambar menggunakan model TF-Lite, TFJS, dan SavedModel.
  
# Persyaratan
File requirements.txt berisi pustaka yang digunakan dalam proyek:

tensorflow==2.18.0
numpy
matplotlib
scikit-learn
tensorflowjs

Hasil
1. Akurasi Model:
   - Akurasi pada training set: 85%+
   - Akurasi pada test set: 85%+
2. Visualisasi:
   - Plot akurasi dan loss disediakan untuk mengevaluasi proses pelatihan.
3. Inferensi:
   - Model berhasil melakukan inferensi pada gambar baru.

Contoh hasil:

Gambar: Mobil → Prediksi: Mobil

Gambar: Pesawat → Prediksi: Pesawat

Cara Menjalankan
Persiapan Lingkungan:
- Instal pustaka yang diperlukan dari requirements.txt.
- Muat dataset CIFAR-10 dan jalankan notebook notebook.ipynb.

Inferensi:
- Gunakan model SavedModel, TF-Lite, atau TFJS untuk melakukan inferensi pada gambar baru.

Deployment:
- Deploy model SavedModel di server atau cloud.
- Gunakan model TF-Lite di perangkat mobile.
- Gunakan model TFJS di browser atau aplikasi berbasis JavaScript.
