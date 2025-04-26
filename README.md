# Image Classification Workflow Using CIFAR-10 Dataset

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan klasifikasi gambar pada dataset CIFAR-10 menggunakan pendekatan Convolutional Neural Network (CNN). Alur kerja mencakup:
- Preprocessing dan augmentasi data
- Pembuatan arsitektur CNN
- Fine-tuning model pra-latih MobileNetV2
- Pelatihan model menggunakan callback
- Evaluasi dan visualisasi hasil
- Penyimpanan model dalam format SavedModel, TF-Lite, dan TensorFlow.js

## Dataset
Dataset CIFAR-10 terdiri dari 60.000 gambar berukuran 32x32 piksel dengan 10 kelas: pesawat, mobil, burung, kucing, rusa, anjing, katak, kuda, kapal, truk.
```
# Muat Dataset CIFAR-10
# Dataset CIFAR-10 terdiri dari 50.000 gambar untuk pelatihan dan 10.000 gambar untuk pengujian.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

## Hasil Latih Model
- **Training Accuracy**: 92.41%
- **Test Accuracy**: 86.73%

## Requirements
Project ini menggunakan library Python berikut:
- TensorFlow
- Numpy
- Matplotlib
- OpenCV
- Scikit-learn
- TensorFlow.js

## File Penyimpanan Model
Model disimpan dalam tiga format:
1. **SavedModel**: untuk deployment berbasis TensorFlow.
2. **TF-Lite**: untuk aplikasi perangkat dengan resource terbatas.
3. **TensorFlow.js**: untuk aplikasi berbasis web.

## Output
- Model terlatih
- Grafik visualisasi akurasi dan loss
- Model dalam format SavedModel, TF-Lite, dan TensorFlow.js

