import os
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import numpy as np

BASE_DIR = 'dataset_ikan/train'
CLASS_NAMES = ['gerabah', 'kembung', 'tongkol']

# Buat figure untuk plot dengan ukuran yang lebih kecil (misal 6x7)
fig, axes = plt.subplots(3, 2, figsize=(6, 7))
fig.suptitle('Perbandingan Gambar Asli vs Setelah Preprocessing', fontsize=12, y=0.98)

for i, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(BASE_DIR, class_name)
    
    # Cek apakah folder kelas ada dan ada isinya
    if not os.path.exists(class_path):
        print(f"Folder {class_path} tidak ditemukan.")
        continue
        
    image_files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        print(f"Tidak ada gambar di folder {class_name}.")
        continue
        
    # Ambil 1 gambar secara acak (atau gambar pertama)
    sample_image_file = image_files[0] 
    img_path = os.path.join(class_path, sample_image_file)
    
    # --- 1. Gambar Asli ---
    img_original = mpimg.imread(img_path)
    axes[i, 0].imshow(img_original)
    axes[i, 0].set_title(f"Asli - {class_name.capitalize()}\nUkuran: {img_original.shape}")
    axes[i, 0].axis('off')
    
    # --- 2. Gambar Setelah Preprocessing ---
    # Load gambar dengan target size 150x150 seperti saat di CNN
    img_preprocessed = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img_preprocessed)
    
    # Normalisasi (1./255)
    img_array_normalized = img_array / 255.0
    
    axes[i, 1].imshow(img_array_normalized)
    axes[i, 1].set_title(f"Preprocessed - {class_name.capitalize()}\nUkuran: (150, 150, 3)")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.9) # Memberi ruang untuk judul utama
plt.savefig('sample_preprocessing.png', dpi=300)
print("Berhasil! Gambar perbandingan telah disimpan sebagai 'sample_preprocessing.png'")
