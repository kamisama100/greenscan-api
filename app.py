from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from tensorflow import keras

IMG_SIZE = 224
model = keras.models.load_model('plant_classifier_model.keras')
with open('class_names.txt') as f:
    class_names = f.read().splitlines()

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Accept up to 4 images with keys: image1, image2, image3, image4 (or just image for single)
    image_keys = ['image', 'image1', 'image2', 'image3', 'image4']
    results = []
    for key in image_keys:
        if key in request.files:
            file = request.files[key]
            # Check file extension
            if not (file.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
                results.append({'error': f'Unsupported file type for {file.filename}'})
                continue
            try:
                img = keras.preprocessing.image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
                img_array = keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = model.predict(img_array)
                predicted_class = class_names[np.argmax(predictions[0])]
                confidence = float(np.max(predictions[0]))
                results.append({'class': predicted_class, 'confidence': confidence, 'filename': file.filename})
            except Exception as e:
                results.append({'error': str(e), 'filename': file.filename})
    if not results:
        return jsonify({'error': 'No valid images provided. Please upload 1-4 images with keys image, image1, image2, image3, or image4.'}), 400
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)