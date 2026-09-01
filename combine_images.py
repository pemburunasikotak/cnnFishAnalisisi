import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# List gambar beserta judulnya
image_files = [
    ('dataset_distribution.png', '1. Dataset Overview'),
    ('sample_preprocessing.png', '2. Image Preprocessing'),
    ('data_augmentation.png', '3. Data Augmentation'),
    ('learning_curves.png', '4. Training Process'),
    ('model_evaluation.png', '5. Model Evaluation'),
    ('confusion_matrix.png', '6. Confusion Matrix Analysis'),
    ('roc_curve.png', '7. ROC Curve and AUC'),
    ('tflite_evaluation.png', '8. TensorFlow Lite Evaluation')
]

# Set ukuran kanvas besar
fig, axes = plt.subplots(4, 2, figsize=(20, 28))
fig.suptitle('Ringkasan Evaluasi Model CNN (Prediksi Ikan)', fontsize=32, fontweight='bold', y=0.98)

# Flatten axes agar mudah diiterasi
axes = axes.flatten()

# Loop untuk memasukkan setiap gambar ke subplot
for i, (filename, title) in enumerate(image_files):
    if os.path.exists(filename):
        img = mpimg.imread(filename)
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=20, pad=15)
        axes[i].axis('off')
    else:
        axes[i].text(0.5, 0.5, f"Image {filename} not found", ha='center', va='center', fontsize=15)
        axes[i].axis('off')

# Semua subplot terisi

plt.tight_layout(pad=3.0)
plt.subplots_adjust(top=0.95)

# Simpan hasil akhir
output_filename = 'combined_evaluation_summary.png'
plt.savefig(output_filename, dpi=300)
print(f"Berhasil menggabungkan gambar ke {output_filename}")
