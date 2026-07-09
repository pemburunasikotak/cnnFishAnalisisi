import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import numpy as np
import os
import sys

# Nama file model yang sudah disimpan
MODEL_PATH = 'model_prediksi_ikan.h5'

# Daftar kelas (Pastikan urutannya sama dengan saat training di folder dataset_real/train)
# Anda bisa mengecek folder train untuk memastikan urutan abjadnya.
CLASS_NAMES = ['Gerabah tidak segar', 'Kembung tidak segar', 'Kuniran tidak segar', 'gerabah segar', 'kembung segar', 'kuniran segar']

def predict_image(img_path):
    # Cek apakah file model ada
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' tidak ditemukan.")
        print("Pastikan Anda sudah menjalankan 'train_cnn.py' sampai selesai untuk melatih dan menyimpan model.")
        return
        
    # Cek apakah file gambar ada
    if not os.path.exists(img_path):
        print(f"Error: Gambar '{img_path}' tidak ditemukan.")
        return

    # Load model
    print("Memuat model 'model_prediksi_ikan.h5'...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Proses gambar agar sesuai dengan input MobileNetV2 (224x224)
    print(f"Memproses gambar: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) # Menambahkan dimensi batch (jadi (1, 224, 224, 3))
    
    # Normalisasi menggunakan fungsi bawaan MobileNetV2
    img_array = preprocess_input(img_array)
    
    # Lakukan Prediksi
    predictions = model.predict(img_array)
    
    # Ambil indeks dengan nilai probabilitas tertinggi
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence = predictions[0][predicted_class_index] * 100
    
    # Tampilkan Hasil Visualisasi di Terminal
    print("\n" + "="*40)
    print("            HASIL PREDIKSI            ")
    print("="*40)
    print(f"Jenis Ikan     : {predicted_class_name.upper()}")
    print(f"Tingkat Yakin  : {confidence:.2f}%")
    print("="*40)
    
    print("\nDetail Probabilitas Setiap Kelas:")
    for i, class_name in enumerate(CLASS_NAMES):
        print(f"- {class_name.capitalize()}: {predictions[0][i] * 100:.2f}%")

if __name__ == "__main__":
    # Script ini dapat menerima path gambar dari terminal
    if len(sys.argv) > 1:
        target_image_path = sys.argv[1]
        predict_image(target_image_path)
    else:
        print("\nCara Penggunaan: python test_cnn.py <path_gambar_ikan_anda>")
        print("Contoh: python test_cnn.py dataset_real/validation/kembung_segar/gambar_01.jpg\n")
