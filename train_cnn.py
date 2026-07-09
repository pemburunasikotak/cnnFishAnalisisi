import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os

# 1. Konfigurasi Dataset dan Direktori
BASE_DIR = 'dataset_real'
TRAIN_DIR = os.path.join(BASE_DIR, 'train')
VALIDATION_DIR = os.path.join(BASE_DIR, 'validation')

# Menggunakan 224x224 yang merupakan standar optimal untuk MobileNetV2
IMG_WIDTH, IMG_HEIGHT = 224, 224
BATCH_SIZE = 32
EPOCHS = 30 # Dapat ditingkatkan karena menggunakan early stopping

# 2. Data Augmentation & Preprocessing
# MobileNetV2 memiliki rentang input dari -1 hingga 1, kita gunakan fungsi preprocess_input bawaannya.
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)

# --- Menghitung Jumlah Data Per Kelas ---
print("\n" + "="*40)
print("JUMLAH DATA LATIH (TRAIN) PER KELAS:")
print("="*40)
if train_generator.samples > 0:
    train_classes, train_counts = np.unique(train_generator.classes, return_counts=True)
    for cls_idx, count in zip(train_classes, train_counts):
        class_name = list(train_generator.class_indices.keys())[list(train_generator.class_indices.values()).index(cls_idx)]
        print(f"- {class_name.capitalize()}: {count} gambar")
else:
    print("Tidak ada gambar.")

print("\n" + "="*40)
print("JUMLAH DATA VALIDASI PER KELAS:")
print("="*40)
if validation_generator.samples > 0:
    val_classes, val_counts = np.unique(validation_generator.classes, return_counts=True)
    for cls_idx, count in zip(val_classes, val_counts):
        class_name = list(validation_generator.class_indices.keys())[list(validation_generator.class_indices.values()).index(cls_idx)]
        print(f"- {class_name.capitalize()}: {count} gambar")
else:
    print("Tidak ada gambar.")
print("="*40 + "\n")
# ---------------------------------------

if train_generator.samples == 0:
    print("\n[ERROR] Tidak ada gambar di folder train! Silakan masukkan gambar ke dataset_ikan/train/...")
    exit()

# 3. Membangun Arsitektur CNN menggunakan Transfer Learning (MobileNetV2)
print("\nMemuat arsitektur MobileNetV2...")
base_model = MobileNetV2(
    weights='imagenet', 
    include_top=False, 
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Membekukan (Freeze) bobot pre-trained base model untuk iterasi pertama
base_model.trainable = False

# Menambahkan custom layer untuk klasifikasi
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Kompilasi model
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              metrics=['accuracy'])

model.summary()

# 4. Melatih Model dengan Callbacks (Early Stopping & Reduce LR)
print("Memulai pelatihan model...")
has_validation = validation_generator.samples > 0

# Callbacks untuk mencegah overfitting dan mengoptimalkan learning rate
callbacks = [
    EarlyStopping(monitor='val_loss' if has_validation else 'loss', patience=7, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss' if has_validation else 'loss', factor=0.5, patience=3, verbose=1, min_lr=1e-6)
]

fit_kwargs = {
    'epochs': EPOCHS,
    'callbacks': callbacks
}

if has_validation:
    fit_kwargs['validation_data'] = validation_generator

history = model.fit(
    train_generator,
    **fit_kwargs
)

# Fine-tuning: Unfreeze beberapa layer atas dari base_model
print("\nMemulai fase Fine-Tuning...")
base_model.trainable = True
# Bekukan 100 layer pertama, latih sisanya
for layer in base_model.layers[:100]:
    layer.trainable = False

# Kompilasi ulang dengan learning rate yang lebih kecil
model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              metrics=['accuracy'])

# Melatih lagi untuk fine-tuning
history_finetune = model.fit(
    train_generator,
    epochs=10, # Tambahan epoch untuk fine-tuning
    validation_data=validation_generator if has_validation else None,
    callbacks=[
        EarlyStopping(monitor='val_loss' if has_validation else 'loss', patience=5, restore_best_weights=True, verbose=1)
    ]
)

# Menyimpan model akhir
model.save('model_prediksi_ikan.h5')
print("\nModel berhasil disimpan sebagai 'model_prediksi_ikan.h5'")

# 5. Membuat Grafik Performa (Learning Curves - Gabungan)
# Menggabungkan history
acc = history.history['accuracy'] + history_finetune.history['accuracy']
loss = history.history['loss'] + history_finetune.history['loss']
if has_validation:
    val_acc = history.history['val_accuracy'] + history_finetune.history['val_accuracy']
    val_loss = history.history['val_loss'] + history_finetune.history['val_loss']

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
if has_validation:
    plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(len(history.history['accuracy'])-1, color='red', linestyle='--', label='Start Fine Tuning')
plt.title('Akurasi Training & Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Akurasi')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
if has_validation:
    plt.plot(val_loss, label='Validation Loss')
plt.axvline(len(history.history['loss'])-1, color='red', linestyle='--', label='Start Fine Tuning')
plt.title('Loss Training & Fine-tuning')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curves.png')
print("Grafik disimpan sebagai 'learning_curves.png'")

# 6. Evaluasi dan Confusion Matrix HANYA jika ada data validasi
if has_validation:
    print("\nMelakukan prediksi pada data validasi...")
    validation_generator.reset()
    Y_pred = model.predict(validation_generator)
    y_pred = np.argmax(Y_pred, axis=1)

    y_true = validation_generator.classes
    class_names = list(validation_generator.class_indices.keys())

    # 7. Membuat Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='gray')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Label Sebenarnya', fontsize=11, labelpad=10)
    plt.xlabel('Label Prediksi', fontsize=11, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    print("Confusion matrix disimpan sebagai 'confusion_matrix.png'")

    # 8. Menampilkan Tabel Metrik Evaluasi
    print("\n=== Laporan Klasifikasi (Metrik Evaluasi) ===")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    with open('evaluation_metrics.txt', 'w') as f:
        f.write("=== Laporan Klasifikasi (Metrik Evaluasi) ===\n")
        f.write(report)
    print("Metrik evaluasi disimpan ke 'evaluation_metrics.txt'")
else:
    print("\n[INFO] Folder validation kosong, sehingga langkah Evaluasi dan Confusion Matrix dilewati.")
