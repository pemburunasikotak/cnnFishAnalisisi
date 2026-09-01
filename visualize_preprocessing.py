import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import numpy as np

img_path = '/Users/mbar/Documents/coding/kampus/cnnPrediksiDipa/dataset_real/train/gerabah segar/1.jpeg'

fig, axes = plt.subplots(1, 2, figsize=(8, 4))
fig.suptitle('Perbandingan Gambar Asli vs Setelah Preprocessing', fontsize=14, y=1.05)

if os.path.exists(img_path):
    # --- 1. Gambar Asli ---
    img_original = mpimg.imread(img_path)
    axes[0].imshow(img_original)
    axes[0].set_title(f"Asli\nUkuran: {img_original.shape}")
    axes[0].axis('off')
    
    # --- 2. Gambar Setelah Preprocessing ---
    # Load gambar dengan target size 224x224 (sesuai update sebelumnya)
    img_preprocessed = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img_preprocessed)
    
    # Normalisasi (1./255)
    img_array_normalized = img_array / 255.0
    
    axes[1].imshow(img_array_normalized)
    axes[1].set_title(f"Preprocessed\nUkuran: (224, 224, 3)")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig('sample_preprocessing.png', dpi=300, bbox_inches='tight')
    print("Berhasil! Gambar perbandingan telah disimpan sebagai 'sample_preprocessing.png'")
else:
    print(f"File tidak ditemukan: {img_path}")
