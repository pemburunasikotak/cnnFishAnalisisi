import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
import matplotlib.pyplot as plt
import os
import time

# 1. Konfigurasi
BASE_DIR = 'dataset_real'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # Epoch disingkat untuk ablation study

# 2. Arsitektur 1: Custom CNN (Baseline Lama)
def build_custom_cnn(num_classes):
    model = Sequential([
        Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        Conv2D(32, (3,3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D(2,2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# 3. Arsitektur 2: MobileNetV2 (SOTA)
def build_mobilenet_v2(num_classes):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    base_model.trainable = False
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_study():
    # Cek jumlah data
    train_datagen_custom = ImageDataGenerator(rescale=1./255)
    train_gen_custom = train_datagen_custom.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    if train_gen_custom.samples == 0:
        print("Data latih tidak ditemukan. Pastikan direktori dataset_real terisi.")
        return
        
    num_classes = len(train_gen_custom.class_indices)
    
    print("\n" + "="*50)
    print("1. MELATIH CUSTOM CNN (BASELINE)")
    print("="*50)
    val_gen_custom = ImageDataGenerator(rescale=1./255).flow_from_directory(VALIDATION_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    
    model_custom = build_custom_cnn(num_classes)
    start_time = time.time()
    hist_custom = model_custom.fit(train_gen_custom, validation_data=val_gen_custom, epochs=EPOCHS, verbose=1)
    time_custom = time.time() - start_time

    print("\n" + "="*50)
    print("2. MELATIH MOBILENET-V2")
    print("="*50)
    train_datagen_mb = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    val_datagen_mb = ImageDataGenerator(preprocessing_function=mobilenet_preprocess)
    
    train_gen_mb = train_datagen_mb.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE)
    val_gen_mb = val_datagen_mb.flow_from_directory(VALIDATION_DIR, target_size=IMG_SIZE, batch_size=BATCH_SIZE)

    model_mb = build_mobilenet_v2(num_classes)
    start_time = time.time()
    hist_mb = model_mb.fit(train_gen_mb, validation_data=val_gen_mb, epochs=EPOCHS, verbose=1)
    time_mb = time.time() - start_time
    
    # Visualisasi Komparasi
    plt.figure(figsize=(14, 6))
    
    # Plot Akurasi Validasi
    plt.subplot(1, 2, 1)
    plt.plot(hist_custom.history['val_accuracy'], label='Custom CNN (val_acc)', linestyle='--')
    plt.plot(hist_mb.history['val_accuracy'], label='MobileNetV2 (val_acc)', linewidth=2)
    plt.title('Perbandingan Akurasi Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Akurasi')
    plt.legend()
    
    # Plot Loss Validasi
    plt.subplot(1, 2, 2)
    plt.plot(hist_custom.history['val_loss'], label='Custom CNN (val_loss)', linestyle='--')
    plt.plot(hist_mb.history['val_loss'], label='MobileNetV2 (val_loss)', linewidth=2)
    plt.title('Perbandingan Loss Validasi')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('ablation_study_results.png')
    print("\nHasil komparasi disimpan sebagai 'ablation_study_results.png'")
    
    print("\nRingkasan Kinerja:")
    print(f"Custom CNN  - Val Acc Terbaik: {max(hist_custom.history['val_accuracy'])*100:.2f}%, Waktu: {time_custom:.1f} detik")
    print(f"MobileNetV2 - Val Acc Terbaik: {max(hist_mb.history['val_accuracy'])*100:.2f}%, Waktu: {time_mb:.1f} detik")

if __name__ == "__main__":
    run_study()
