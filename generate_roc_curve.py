import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

BASE_DIR = 'dataset_real/validation'
CLASS_NAMES = ['Gerabah tidak segar', 'Kembung tidak segar', 'Kuniran tidak segar', 'gerabah segar', 'kembung segar', 'kuniran segar']
MODEL_PATH = 'model_prediksi_ikan.h5'

if not os.path.exists(MODEL_PATH):
    print(f"Error: Model {MODEL_PATH} tidak ditemukan.")
    exit()

# Load model
model = load_model(MODEL_PATH)
print("Model berhasil dimuat.")

# Setup test data generator
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    BASE_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False # Penting agar label sesuai dengan urutan prediksi
)

# Ambil label asli (y_true)
y_true = test_generator.classes
# Binarize label untuk multiclass ROC
y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3, 4, 5])
n_classes = y_true_bin.shape[1]

# Prediksi menggunakan model (y_pred)
print("Memulai prediksi pada test set...")
y_pred = model.predict(test_generator)

# Hitung ROC curve dan ROC area untuk masing-masing kelas
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curve
plt.figure(figsize=(10, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve {CLASS_NAMES[i].capitalize()} (area = {roc_auc[i]:.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig('roc_curve.png', dpi=300)
print("Berhasil menyimpan gambar ROC Curve ke 'roc_curve.png'")
