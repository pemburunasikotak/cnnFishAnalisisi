import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import argparse

# Konfigurasi Model
MODEL_PATH = 'model_prediksi_ikan.h5'
CLASS_NAMES = ['Gerabah tidak segar', 'Kembung tidak segar', 'Kuniran tidak segar', 'gerabah segar', 'kembung segar', 'kuniran segar']

def save_and_display_gradcam(img_path, heatmap, cam_path="gradcam_result.jpg", alpha=0.4):
    # Load original image
    img = cv2.imread(img_path)
    
    if img is None:
        print(f"Error: Tidak dapat membaca gambar {img_path}")
        return

    # Resize heatmap agar ukurannya sama dengan gambar asli
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Konversi heatmap ke format RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # Gabungkan gambar asli dengan heatmap
    superimposed_img = heatmap * alpha + img
    
    # Ambil nilai yang tidak melebihi 255
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    # Simpan hasil heatmap
    cv2.imwrite(cam_path, superimposed_img)
    print(f"Heatmap Grad-CAM berhasil disimpan sebagai: {cam_path}")

    # Tampilkan gambar (Matplotlib)
    try:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
        plt.title("Gambar Asli")
        plt.axis('off')

        plt.subplot(1, 2, 2)
        super_rgb = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
        plt.imshow(super_rgb)
        plt.title("Grad-CAM Heatmap")
        plt.axis('off')
        
        plt.savefig("gradcam_plot.png")
        print("Plot perbandingan disimpan sebagai gradcam_plot.png")
    except Exception as e:
        print(f"Tidak dapat menampilkan plot: {e}")

def run_gradcam(img_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model '{MODEL_PATH}' tidak ditemukan.")
        return
        
    print(f"Memuat model {MODEL_PATH}...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    # Cari base_model MobileNetV2 yang ada di dalam model utama (nested model)
    base_model_layer = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            base_model_layer = layer
            break
            
    if base_model_layer is not None:
        # Gunakan layer 'out_relu' dari MobileNetV2 sebagai layer konvolusi terakhir
        try:
            last_conv_layer = base_model_layer.get_layer('out_relu')
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[last_conv_layer.output, model.output]
            )
            print(f"Menggunakan layer: out_relu dari {base_model_layer.name}")
        except Exception:
            # Fallback: cari layer dengan output 4D di dalam base_model
            last_conv_layer_name = None
            for layer in reversed(base_model_layer.layers):
                try:
                    if len(layer.output.shape) == 4:
                        last_conv_layer_name = layer.name
                        break
                except Exception:
                    continue
            grad_model = tf.keras.models.Model(
                inputs=[model.inputs],
                outputs=[base_model_layer.get_layer(last_conv_layer_name).output, model.output]
            )
            print(f"Menggunakan layer fallback: {last_conv_layer_name}")
    else:
        # Model datar (non-nested): cari layer Conv2D terakhir
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            try:
                if len(layer.output.shape) == 4:
                    last_conv_layer_name = layer.name
                    break
            except Exception:
                continue
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )
        print(f"Menggunakan layer: {last_conv_layer_name}")

    print(f"Memproses gambar: {img_path}")
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    preds = model.predict(img_array)
    pred_index = np.argmax(preds[0])
    print(f"Prediksi: {CLASS_NAMES[pred_index]} ({preds[0][pred_index]*100:.2f}%)")
    
    # Generate heatmap
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds_out = grad_model(img_array)
        class_channel = preds_out[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()

    save_and_display_gradcam(img_path, heatmap)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Grad-CAM Explainability untuk CNN Ikan')
    parser.add_argument('image_path', type=str, help='Path ke gambar uji')
    args = parser.parse_args()
    
    run_gradcam(args.image_path)
