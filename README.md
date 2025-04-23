### 1. **Judul dan Tujuan Proyek**
**Judul**: Klasifikasi Gambar Anjing Menggunakan Dataset Stanford Dogs  
**Tujuan**: Membuat model CNN menggunakan dataset Stanford Dogs untuk mengklasifikasikan gambar anjing berdasarkan breed dengan akurasi minimal 85% pada training dan testing set.

---

### 2. **Pengumpulan dan Persiapan Dataset**
- **Dataset**: Stanford Dogs (link: [TensorFlow Stanford Dogs](https://www.tensorflow.org/datasets/catalog/stanford_dogs)).
- **Kriteria Dataset**: Dataset sudah memenuhi syarat minimal 1000 gambar.
- **Langkah Persiapan**:
  1. Unduh dataset Stanford Dogs dan pastikan formatnya sesuai untuk TensorFlow.
  2. Lakukan **pembagian dataset** menjadi:
     - **Train Set** (misal 70% dari dataset)
     - **Test Set** (20%)
     - **Validation Set** (10%)
  3. Jika diperlukan, gunakan data augmentation untuk meningkatkan variasi dataset.

---

### 3. **Arsitektur Model**
Gunakan library **Keras** untuk membangun model:
- Gunakan **Sequential Model**.
- Tambahkan lapisan berikut:
  - **Conv2D**: Untuk ekstraksi fitur.
  - **Pooling Layer**: Untuk pengurangan dimensionalitas dan fokus fitur penting.
  - **Dense Layer**: Untuk klasifikasi.

Contoh kode dasar:
```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(120, activation='softmax')  # Sesuaikan jumlah kelas
])
```

---

### 4. **Training Model**
- **Callback**: Gunakan callback seperti EarlyStopping dan ModelCheckpoint.
- **Optimizer**: Gunakan optimizer seperti Adam.
- **Loss Function**: Cross-entropy (categorical_crossentropy).
- **Batch Size dan Epoch**: Sesuaikan untuk memastikan model mencapai akurasi minimal 85%.

Contoh:
```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),
    ModelCheckpoint('best_model.h5', save_best_only=True)
]

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(train_data, epochs=20, batch_size=32, validation_data=val_data, callbacks=callbacks)
```

---

### 5. **Evaluasi Model**
- **Akurasi dan Loss**: Evaluasi model pada testing set untuk memastikan akurasi minimal 85%.
- **Visualisasi**: Plot akurasi dan loss untuk training dan validation.
Contoh:
```python
import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()
```

---

### 6. **Simpan Model**
Simpan model dalam berbagai format:
- **SavedModel**:
  ```python
  model.save('saved_model')
  ```
- **TF-Lite**:
  ```python
  import tensorflow as tf
  converter = tf.lite.TFLiteConverter.from_saved_model('saved_model')
  tflite_model = converter.convert()
  with open('model.tflite', 'wb') as f:
      f.write(tflite_model)
  ```
- **TFJS**:
  ```python
  !tensorflowjs_converter --input_format=tf_saved_model --output_node_names='output_node' saved_model tfjs_model
  ```

---

### 7. **Inference dan Dokumentasi**
- **Inference**: Gunakan salah satu format model (SavedModel, TF-Lite, atau TFJS) untuk melakukan inferensi.
- **Bukti Inferensi**: Simpan hasil inferensi dalam bentuk screenshot atau output di notebook.
- **Dokumentasi**: Pastikan seluruh proses proyek terdokumentasi dengan baik, termasuk langkah-langkah preprocessing, hasil visualisasi, dan kode inferensi.

---
