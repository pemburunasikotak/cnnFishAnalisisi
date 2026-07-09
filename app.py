from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from werkzeug.utils import secure_filename
from flasgger import Swagger

app = Flask(__name__)
swagger = Swagger(app, template={
    "info": {
        "title": "API Prediksi Kesegaran Ikan (CNN MobileNetV2)",
        "description": "API untuk memprediksi jenis dan tingkat kesegaran ikan menggunakan model Deep Learning berarsitektur MobileNetV2.",
        "version": "1.0.0"
    }
})

# Konfigurasi folder upload dan tipe file
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 # Max 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Path model
MODEL_PATH = 'model_prediksi_ikan.h5'
CLASS_NAMES = ['Gerabah tidak segar', 'Kembung tidak segar', 'Kuniran tidak segar', 'gerabah segar', 'kembung segar', 'kuniran segar']

# Muat model secara global agar tidak me-load ulang setiap kali prediksi
print("Memuat model CNN...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model berhasil dimuat!")
except Exception as e:
    print(f"Error memuat model: {e}")
    model = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediksi Jenis dan Kesegaran Ikan
    Upload gambar ikan (.jpg, .jpeg, .png, .webp) untuk mendapatkan hasil prediksi dari model CNN.
    ---
    tags:
      - Prediksi
    consumes:
      - multipart/form-data
    parameters:
      - name: file
        in: formData
        type: file
        required: true
        description: Gambar ikan yang akan diprediksi
    responses:
      200:
        description: Berhasil melakukan prediksi
        schema:
          type: object
          properties:
            success:
              type: boolean
              example: true
            prediction:
              type: string
              example: Kembung Segar
            confidence:
              type: number
              example: 94.5
            details:
              type: array
              items:
                type: object
                properties:
                  class:
                    type: string
                  probability:
                    type: number
      400:
        description: Bad Request (File tidak ada atau format salah)
      500:
        description: Internal Server Error (Model belum dilatih atau error sistem)
    """
    if model is None:
        return jsonify({'error': 'Model tidak ditemukan. Pastikan Anda sudah melatih model terlebih dahulu.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Tidak ada file yang diunggah'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'File belum dipilih'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocessing gambar
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            
            # Normalisasi khusus MobileNetV2
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            img_array = preprocess_input(img_array)
            
            # Prediksi
            predictions = model.predict(img_array)
            predicted_class_index = np.argmax(predictions[0])
            predicted_class_name = CLASS_NAMES[predicted_class_index].capitalize()
            confidence = float(predictions[0][predicted_class_index] * 100)
            
            # Detail semua probabilitas
            details = []
            for i, class_name in enumerate(CLASS_NAMES):
                details.append({
                    'class': class_name.capitalize(),
                    'probability': float(predictions[0][i] * 100)
                })
            
            # Hapus file setelah diprediksi agar storage tidak penuh
            os.remove(filepath)
            
            return jsonify({
                'success': True,
                'prediction': predicted_class_name,
                'confidence': round(confidence, 2),
                'details': details
            })
            
        except Exception as e:
            return jsonify({'error': f'Terjadi kesalahan saat memproses gambar: {str(e)}'}), 500
            
    return jsonify({'error': 'Tipe file tidak didukung. Gunakan JPG, JPEG, atau PNG'}), 400

if __name__ == '__main__':
    # host='0.0.0.0' mengizinkan akses dari jaringan eksternal (seperti device Android Anda)
    app.run(host='0.0.0.0', debug=True, port=5001)
