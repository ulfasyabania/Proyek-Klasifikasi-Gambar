# Image Classification Workflow Using CIFAR-10 Dataset

## Deskripsi Proyek
Proyek ini bertujuan untuk melakukan klasifikasi gambar pada dataset CIFAR-10 menggunakan pendekatan Convolutional Neural Network (CNN). Alur kerja mencakup:
- Preprocessing dan augmentasi data
- Pembuatan arsitektur CNN
- Fine-tuning model pra-latih MobileNetV2
- Pelatihan model menggunakan callback
- Evaluasi dan visualisasi hasil
- Penyimpanan model dalam format SavedModel, TF-Lite, dan TensorFlow.js

## Import dan Setup

1. **Import Library dan Modul**: Memuat library dan modul yang akan digunakan dalam proses pengolahan data dan pelatihan model. Dalam hal ini, library seperti `tensorflow`, `numpy`, `matplotlib`, dan `cv2` (OpenCV) diimpor untuk berbagai keperluan.
   
2. **Persiapan Alat dan Parameter**: Mempersiapkan fungsi, kelas, atau komponen lain, seperti dataset (`cifar10`), preprocessing (`ImageDataGenerator`), dan arsitektur model (`Sequential`, `Conv2D`, dll.).
```
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2  # # OpenCV digunakan untuk transformasi khusus
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
```
**Insight Tambahan**:
- Tahapan ini penting untuk memastikan bahwa semua dependensi telah siap sebelum memulai langkah selanjutnya dalam workflow, seperti memuat data atau pelatihan model.
- Dengan mengorganisasikan semua impor dalam satu bagian di awal, membuat kode lebih terstruktur dan mudah dibaca.

## Muat Dataset dan Normalisasi

1. **Muat Dataset CIFAR-10**:
   - Dataset CIFAR-10 terdiri dari 50.000 gambar untuk pelatihan dan 10.000 gambar untuk pengujian.
   - Dataset ini memuat gambar dengan resolusi 32x32 piksel, dikelompokkan ke dalam 10 kelas, seperti pesawat, mobil, burung, dll.

2. **Normalisasi Data**:
   - Nilai piksel setiap gambar dikonversi dari rentang [0,255] ke [0,1] dengan membaginya dengan 255.0. 
   - Tujuannya adalah mempercepat proses pelatihan model dan menghindari masalah numerik yang dapat muncul dari nilai piksel besar.

## **"Pembagian Data dan Pra-pemrosesan Label"**

1. **Pembagian Data**:
   - Menggunakan `train_test_split` untuk membagi 20% data pelatihan ke dalam set validasi. Ini penting untuk mengevaluasi performa model secara obyektif selama pelatihan.
   - Parameter `random_state=42` memastikan bahwa pembagian data dilakukan secara deterministik, sehingga menghasilkan output yang konsisten.

2. **Konversi Label ke One-Hot Encoding**:
   - Fungsi `to_categorical` digunakan untuk mengonversi label ke representasi one-hot encoding, yang merupakan format yang umum digunakan dalam pembelajaran mendalam. Misalnya, label `3` dikonversi menjadi `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]` untuk 10 kelas.

3. **Verifikasi Bentuk Data (Shapes)**:
   - Mencetak bentuk (shape) dari label pelatihan (`y_train`), validasi (`y_val`), dan pengujian (`y_test`) untuk memastikan bahwa data diorganisir dengan benar. Output akan menunjukkan:
     - `y_train shape: (40000, 10)`
     - `y_val shape: (10000, 10)`
     - `y_test shape: (10000, 10)`

**Insight Tambahan**:
- Langkah ini membantu memvalidasi bahwa data pelatihan dan validasi telah dipisahkan secara proporsional, dan label sudah dalam format yang tepat untuk digunakan dalam pelatihan model.
- Dengan one-hot encoding, model memiliki kemampuan untuk belajar dari distribusi kelas secara langsung.

## **Transformasi Resolusi Acak**

1. **Tujuan Fungsi**:
   - Fungsi ini dirancang untuk mensimulasikan gambar dengan resolusi tidak seragam, kemudian memastikan ukuran gambar kembali ke standar 32x32 menggunakan padding atau cropping. Hal ini berguna untuk menangani data dengan resolusi variatif.

2. **Proses Transformasi**:
   - **Konversi Format**: Gambar awal dalam format float32 dengan rentang nilai [0,1] diubah ke uint8 dengan rentang [0,255]. Format ini diperlukan agar kompatibel dengan operasi OpenCV.
   - **Skala Acak**: Ukuran gambar dimodifikasi secara acak dalam kisaran 80% hingga 120% dari ukuran asli (32x32 piksel).
   - **Penyesuaian Ukuran**:
     - Jika gambar lebih kecil dari 32x32, padding ditambahkan menggunakan metode `cv2.copyMakeBorder`.
     - Jika gambar lebih besar dari 32x32, cropping dilakukan untuk mengembalikan ukuran ke standar.

3. **Output**:
   - Setelah transformasi, gambar dikembalikan dalam format float32 dengan rentang nilai [0,1], siap digunakan dalam model pembelajaran mendalam.

**Insight Tambahan**:
- Langkah ini membantu meningkatkan keragaman data pelatihan, sehingga model lebih robust terhadap variasi resolusi di data dunia nyata.
- Pemrosesan padding menggunakan `cv2.BORDER_REFLECT` menciptakan efek alami pada tepi gambar, mengurangi artefak.

## **Augmentasi Data dan Pembuatan Generator Batch**

1. **Augmentasi Data untuk Pelatihan**:
   - Menggunakan `ImageDataGenerator` untuk meningkatkan variasi data pelatihan dengan augmentasi, seperti:
     - *Transformasi resolusi tidak seragam*: Menggunakan fungsi kustom `random_resolution_transform`.
     - *Rotasi*: Gambar dapat diputar hingga 15 derajat.
     - *Perpindahan*: Gambar dapat bergeser horizontal dan vertikal hingga 10% dari dimensi aslinya.
     - *Flip Horizontal*: Gambar bisa dibalik secara horizontal untuk memperkaya data.

2. **Generator Data Pelatihan**:
   - `train_datagen.flow` menghasilkan batch data pelatihan secara otomatis dengan ukuran batch 64. Ini sangat efisien dalam memproses data dalam jumlah besar saat pelatihan model.

3. **Generator Data Validasi**:
   - Data validasi dibuat menggunakan `ImageDataGenerator`, tetapi tanpa augmentasi, untuk memastikan evaluasi model dilakukan pada data yang tidak dimodifikasi.

**Manfaat**:
- Augmentasi data membantu mencegah overfitting dengan memperluas keragaman data pelatihan.
- Menggunakan generator batch memungkinkan pelatihan model pada dataset besar tanpa membebani memori.

## **Load Model Pre-trained dan Konfigurasi Fine-Tuning**

1. **Muat Model Pra-latih**:
   - `MobileNetV2` dimuat dengan bobot yang sudah dilatih sebelumnya menggunakan dataset ImageNet (`weights='imagenet'`).
   - Parameter `include_top=False` memastikan bagian fully connected (klasifikasi asli) dihilangkan sehingga model hanya digunakan sebagai ekstraktor fitur.
   - Ukuran input disesuaikan menjadi `(32, 32, 3)` untuk mencocokkan dimensi gambar dataset CIFAR-10.

2. **Freeze Lapisan untuk Transfer Learning**:
   - Semua lapisan kecuali 20 lapisan terakhir pada `base_model` dibekukan (`layer.trainable = False`). Artinya, bobot lapisan ini tidak akan diperbarui selama pelatihan.
   - Ini adalah teknik transfer learning yang memanfaatkan fitur yang telah dipelajari dari model pra-latih, sambil memungkinkan beberapa lapisan akhir untuk beradaptasi dengan tugas baru (fine-tuning).

**Manfaat Langkah Ini**:
- Mengurangi waktu pelatihan karena sebagian besar lapisan tidak perlu dilatih.
- Memanfaatkan keahlian dari model pra-latih yang sudah memahami fitur-fitur umum gambar.

## **Bangun Model CNN dengan Arsitektur Berlapis**

1. **Arsitektur Berlapis**:
   - Membangun model secara berurutan (Sequential), dengan arsitektur yang terdiri dari blok lapisan berulang:
     - *Blok 1*: Lapisan konvolusi dengan 64 filter, diikuti oleh normalisasi batch, pooling, dan dropout untuk mencegah overfitting.
     - *Blok 2*: Mirip dengan Blok 1 tetapi dengan 128 filter, memungkinkan ekstraksi fitur yang lebih kompleks.
     - *Blok 3*: Lapisan konvolusi dengan 256 filter untuk mengolah fitur tingkat tinggi, diakhiri dengan pooling global untuk agregasi fitur secara keseluruhan.

2. **Lapisan Fully Connected**:
   - Lapisan Dense dengan 512 unit memberikan kemampuan pengambilan keputusan yang lebih kompleks.
   - Dropout dengan nilai 0.5 membantu mengurangi overfitting pada tahap klasifikasi.
   - Lapisan Dense terakhir dengan aktivasi `softmax` digunakan untuk klasifikasi ke 10 kelas (dataset CIFAR-10).

3. **Fitur Tambahan**:
   - *Batch Normalization*: Menstabilkan distribusi data antar lapisan, mempercepat pelatihan dan meningkatkan performa.
   - *GlobalAveragePooling2D*: Mencegah pengurangan dimensi data secara berlebihan, mempertahankan informasi spasial untuk klasifikasi akhir.

**Manfaat Arsitektur Ini**:
- Dengan kombinasi konvolusi, pooling, dan fully connected layers, model mampu menangkap pola dan fitur dari data gambar.
- Dropout dan batch normalization membuat model lebih robust terhadap data dunia nyata.

## **Kompilasi Model**

1. **Pengaturan Optimizer**:
   - `optimizer='adam'`: Adam adalah algoritma optimasi yang efisien dan adaptif, sering digunakan dalam pembelajaran mendalam karena kemampuannya menyesuaikan learning rate secara dinamis.

2. **Definisi Loss Function**:
   - `loss='categorical_crossentropy'`: Fungsi loss ini digunakan untuk tugas klasifikasi multikelas, seperti CIFAR-10, yang memiliki 10 kelas. Ini membantu mengukur seberapa baik model memprediksi probabilitas kelas yang benar.

3. **Metode Evaluasi**:
   - `metrics=['accuracy']`: Metode untuk mengevaluasi performa model selama pelatihan dan validasi dengan mengukur tingkat akurasi prediksi.

**Manfaat Pengaturan Ini**:
- Kombinasi Adam optimizer dan categorical crossentropy loss memberikan keseimbangan yang baik antara kecepatan konvergensi dan akurasi prediksi.
- Akurasi sebagai metrik memungkinkan Anda memantau kemajuan pelatihan model secara mudah.

## **Pengaturan Learning Rate dengan Callback Scheduler**

1. **Definisi Fungsi Learning Rate**:
   - `lr_schedule(epoch)`: Mengembalikan nilai learning rate yang secara bertahap menurun sebesar 5% setiap epoch. Ini membantu model mengonvergensi lebih baik dengan menyesuaikan langkah optimasi selama pelatihan.

2. **Callback LearningRateScheduler**:
   - `LearningRateScheduler(lr_schedule)`: Callback ini digunakan untuk mengatur learning rate sesuai dengan fungsi `lr_schedule` pada setiap epoch. Fungsi ini memberikan kontrol yang fleksibel atas learning rate tanpa harus menghentikan dan mengulang pelatihan model.

**Keunggulan**:
- Teknik pengurangan learning rate bertahap membantu model tetap belajar dari data baru sekaligus mencegah overshooting (lonjakan error karena langkah terlalu besar).
- Callback memungkinkan automasi pengaturan learning rate tanpa perlu intervensi manual.

## **Augmentasi Data dan Pembuatan Generator Batch**

1. **Augmentasi Data Pelatihan**:
   - Anda menggunakan `ImageDataGenerator` untuk menghasilkan data baru dengan berbagai transformasi, seperti:
     - *Rotasi*: Rotasi gambar hingga ±15 derajat.
     - *Geser (Shift)*: Pergeseran gambar horizontal dan vertikal hingga 10% dari dimensi gambar.
     - *Flip Horizontal*: Membalik gambar secara horizontal, meningkatkan keragaman data.

2. **Pembuatan Generator Batch**:
   - `train_generator`: Membuat batch data pelatihan secara otomatis dengan augmentasi yang telah didefinisikan, dengan ukuran batch 64.
   - `val_generator`: Digunakan untuk data validasi tanpa augmentasi. Hal ini memastikan bahwa performa model dievaluasi pada data yang tidak mengalami modifikasi.

**Kegunaan dan Manfaat**:
- **Augmentasi Data**: Membantu model menjadi lebih robust terhadap variasi pada data pelatihan, sehingga performa di dunia nyata meningkat.
- **Efisiensi Memori**: Dengan menggunakan generator batch, dapat menangani dataset yang besar tanpa membebani memori perangkat.

## **Pelatihan Model dengan Callback dan Evaluasi**

1. **Callback untuk Optimasi Pelatihan**:
   - **EarlyStopping**: Menghentikan pelatihan lebih awal jika akurasi validasi tidak meningkat setelah 10 epoch berturut-turut, sambil mengembalikan bobot terbaik (`restore_best_weights=True`).
   - **ReduceLROnPlateau**: Menurunkan learning rate sebesar 50% jika loss validasi tidak membaik setelah 5 epoch. Hal ini membantu model keluar dari kebuntuan atau memperbaiki konvergensi.
   - **LearningRateScheduler**: Mengurangi learning rate secara bertahap sebesar 5% setiap epoch menggunakan fungsi `lr_schedule`.

2. **Pelatihan Model**:
   - Fungsi `model.fit()` menjalankan pelatihan selama 25 epoch, menggunakan generator batch untuk set pelatihan dan validasi. Callback memastikan pelatihan lebih efisien dan stabil.

3. **Evaluasi Performa**:
   - Setelah pelatihan selesai, `model.evaluate()` digunakan untuk mengukur loss dan akurasi pada set pelatihan dan pengujian. Hasil akurasi dicetak dalam persentase untuk memudahkan interpretasi.

**Manfaat Tahapan Ini**:
- Callback meningkatkan efisiensi pelatihan dengan mengoptimalkan learning rate dan mencegah overfitting.
- Evaluasi memastikan bahwa model dapat menghasilkan prediksi yang baik tidak hanya pada data pelatihan, tetapi juga pada data pengujian.


## **Hasil Latih Model**
Log ini memberikan gambaran tentang progres pelatihan model selama 25 epoch:

1. **Akurasi dan Loss**:
   - **Akurasi Pelatihan**: Model meningkat secara konsisten, mencapai akurasi akhir sebesar 92,41%.
   - **Akurasi Validasi**: Mencapai puncaknya sebesar 87,12% pada epoch ke-23, menunjukkan generalisasi yang kuat.
   - Nilai loss untuk pelatihan dan validasi secara umum menurun, mencerminkan bahwa model belajar secara efektif.

2. **Penyesuaian Learning Rate**:
   - Learning rate dimulai pada 0.0010 dan secara bertahap berkurang sebesar 5% setiap epoch. Contohnya, pada epoch terakhir, learning rate sekitar 2.9199e-04. Penurunan ini membantu model melakukan konvergensi secara stabil.

3. **Observasi Awal**:
   - Di awal, akurasi validasi (40,08% pada epoch ke-1) secara signifikan lebih rendah dibandingkan dengan akurasi pelatihan (29,92%), mencerminkan tahap awal model dalam memahami fitur secara efektif.
   - Pada epoch ke-6, akurasi validasi meningkat drastis menjadi 76,11%, menunjukkan peningkatan kemampuan model untuk melakukan generalisasi.

4. **Hasil Akhir**:
   - **Akurasi Pelatihan**: 92,41%
   - **Akurasi Uji**: 86,73%

Hasil ini memvalidasi keandalan alur pelatihan, menyoroti kekuatan augmentasi, arsitektur model, dan pengaturan learning rate yang digunakan.

## **Visualisasi Kinerja Model**

1. **Fungsi Visualisasi**:
   - Menggunakan `matplotlib` untuk membuat visualisasi akurasi dan loss selama pelatihan dan validasi, yang mempermudah analisis kinerja model.

2. **Plot Akurasi**:
   - Pada subplot pertama, grafik akurasi pelatihan dan validasi terhadap epoch diplot.
   - Ini membantu memvisualisasikan bagaimana model belajar dari data pelatihan dan seberapa baik model melakukan generalisasi pada data validasi.

3. **Plot Loss**:
   - Subplot kedua menunjukkan grafik loss pelatihan dan validasi terhadap epoch.
   - Grafik ini memberikan wawasan tentang bagaimana model mengurangi kesalahan selama pelatihan.

4. **Label dan Legenda**:
   - Penggunaan label (seperti *"Akurasi Pelatihan"*, *"Akurasi Validasi"*) serta legend dan judul, menjadikan grafik lebih informatif dan mudah dibaca.

**Manfaat**:
- Grafik ini membantu mengenali potensi overfitting atau underfitting.

## **Simpan dan Konversi Model ke Berbagai Format**

1. **Simpan dalam Format TensorFlow SavedModel**:
   - Fungsi `tf.saved_model.save()` digunakan untuk menyimpan model dalam format TensorFlow SavedModel. Format ini adalah standar yang digunakan untuk deployment model TensorFlow di berbagai platform.
   - Direktori yang dihasilkan (`saved_model_dir`) berisi file konfigurasi, bobot, dan metadata model.

2. **Konversi ke Format TensorFlow Lite (TF-Lite)**:
   - Fungsi `tf.lite.TFLiteConverter.from_saved_model()` digunakan untuk mengonversi model SavedModel ke format yang lebih ringan (TF-Lite), cocok untuk penggunaan di perangkat dengan resource terbatas seperti mobile dan embedded systems.
   - Model yang sudah dikonversi disimpan sebagai file `.tflite`, yang mudah digunakan dalam aplikasi berbasis TF-Lite.

3. **Konversi ke Format TensorFlow.js**:
   - TensorFlow.js memungkinkan untuk menjalankan model pembelajaran mendalam di browser web.
   - Dengan `tensorflowjs_converter`, model SavedModel dikonversi ke format yang kompatibel dengan TensorFlow.js, dan disimpan di direktori `tfjs_model_dir`.

**Kegunaan dan Manfaat**:
- **SavedModel**: Ideal untuk integrasi dengan sistem berbasis TensorFlow.
- **TF-Lite**: Memungkinkan aplikasi efisien pada perangkat dengan sumber daya terbatas.
- **TensorFlow.js**: Mendukung pengembangan aplikasi berbasis web yang interaktif dan ramah pengguna.

## Requirements
Project ini menggunakan library Python berikut:
- TensorFlow
- Numpy
- Matplotlib
- OpenCV
- Scikit-learn
- TensorFlow.js

