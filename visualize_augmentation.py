import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

img_path = '/Users/mbar/Documents/coding/kampus/cnnPrediksiDipa/dataset_real/train/gerabah segar/1.jpeg'

# Image Data Generator with Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

if os.path.exists(img_path):
    # Load and prepare image
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape)
    
    # Plot 1 gambar asli dan 5 gambar augmentasi
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    fig.suptitle('Visualisasi Data Augmentation', fontsize=16)
    
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Asli')
    axes[0, 0].axis('off')
    
    # Generate batches of augmented images
    i = 1
    for batch in datagen.flow(x, batch_size=1):
        row = i // 3
        col = i % 3
        axes[row, col].imshow(image.array_to_img(batch[0]))
        axes[row, col].set_title(f'Augmentasi {i}')
        axes[row, col].axis('off')
        i += 1
        if i > 5:
            break
            
    plt.tight_layout()
    plt.savefig('data_augmentation.png', dpi=300)
    print("Berhasil menyimpan gambar data augmentation ke 'data_augmentation.png'")
else:
    print(f"File tidak ditemukan: {img_path}")
