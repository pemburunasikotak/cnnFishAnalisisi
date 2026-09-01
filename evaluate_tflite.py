import os
import time
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_DIR = 'dataset_real/validation'
H5_MODEL_PATH = 'model_prediksi_ikan.h5'
TFLITE_MODEL_PATH = 'model_prediksi_ikan.tflite'

# 1. Konversi ke TFLite jika belum ada
if not os.path.exists(TFLITE_MODEL_PATH):
    print("Mengonversi model .h5 ke .tflite...")
    model = load_model(H5_MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(TFLITE_MODEL_PATH, 'wb') as f:
        f.write(tflite_model)
    print("Berhasil mengonversi model!")

# 2. Setup Data Generator (ambil beberapa batch saja untuk evaluasi agar cepat)
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=1, # batch size 1 untuk mengukur inference time per gambar
    class_mode='categorical',
    shuffle=False
)

# Ambil 100 gambar pertama untuk evaluasi (atau sebanyak isi dataset test)
num_samples = min(100, test_generator.samples)
print(f"Mengevaluasi {num_samples} gambar...")

# 3. Evaluasi Model .h5
print("Mengevaluasi model H5...")
model_h5 = load_model(H5_MODEL_PATH)

start_time = time.time()
correct_h5 = 0
for i in range(num_samples):
    x, y = test_generator[i]
    pred = model_h5.predict(x, verbose=0)
    if np.argmax(pred) == np.argmax(y):
        correct_h5 += 1
end_time = time.time()

h5_accuracy = correct_h5 / num_samples
h5_time_per_img = (end_time - start_time) / num_samples * 1000 # dalam ms

# 4. Evaluasi Model .tflite
print("Mengevaluasi model TFLite...")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

start_time = time.time()
correct_tflite = 0
for i in range(num_samples):
    x, y = test_generator[i]
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], x)
    # Jalankan inferensi
    interpreter.invoke()
    # Ambil output
    pred = interpreter.get_tensor(output_details[0]['index'])
    
    if np.argmax(pred) == np.argmax(y):
        correct_tflite += 1
end_time = time.time()

tflite_accuracy = correct_tflite / num_samples
tflite_time_per_img = (end_time - start_time) / num_samples * 1000 # dalam ms

# 5. Plot Hasil Perbandingan
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Perbandingan Evaluasi: Keras (.h5) vs TensorFlow Lite (.tflite)')

# Plot Akurasi
models = ['Keras (.h5)', 'TFLite']
acc = [h5_accuracy * 100, tflite_accuracy * 100]
ax1.bar(models, acc, color=['blue', 'green'])
ax1.set_title('Akurasi (%)')
ax1.set_ylim([0, 110])
for i, v in enumerate(acc):
    ax1.text(i, v + 2, f"{v:.2f}%", ha='center')

# Plot Inference Time
times = [h5_time_per_img, tflite_time_per_img]
ax2.bar(models, times, color=['orange', 'red'])
ax2.set_title('Waktu Inferensi per Gambar (ms)')
for i, v in enumerate(times):
    ax2.text(i, v + (max(times)*0.02), f"{v:.2f} ms", ha='center')

plt.tight_layout()
plt.savefig('tflite_evaluation.png', dpi=300)
print("Berhasil menyimpan perbandingan ke 'tflite_evaluation.png'")
