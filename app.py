from flask import Flask, request, send_file, render_template
from flask_cors import CORS
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import io
from PIL import Image
import os

ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}
MODEL_URL = 'https://tfhub.dev/captain-pool/esrgan-tf2/1'

app = Flask(__name__)
CORS(app)
super_res_model = hub.load(MODEL_URL)

@app.route('/')
def index():
    return render_template('index.html')

def process_image(file):
    image = Image.open(file.stream)
    image = np.array(image)
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    super_res_image = super_res_model(image)
    super_res_image = tf.squeeze(super_res_image, axis=0)
    super_res_image = tf.clip_by_value(super_res_image, 0, 255)
    super_res_image = tf.cast(super_res_image, tf.uint8)
    
    super_res_image_pil = Image.fromarray(super_res_image.numpy())
    output_buffer = io.BytesIO()
    super_res_image_pil.save(output_buffer, format="PNG")
    output_buffer.seek(0)

    return output_buffer

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No image uploaded', 400

    file = request.files['file']

    if file.filename == '':
        return 'File name missing', 400

    if file:
        
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in ALLOWED_EXTENSIONS:
            return 'Unsupported image format', 400

        try:
            output_buffer = process_image(file)
            
            return send_file(output_buffer, mimetype='image/png')
        except Exception as e:
            return str(e), 500

    return 'No image uploaded', 400

if __name__ == '__main__':
    app.run(debug=True)