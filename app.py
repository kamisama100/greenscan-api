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
    file = request.files['image']
    img = keras.preprocessing.image.load_img(file, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = float(np.max(predictions[0]))
    return jsonify({'class': predicted_class, 'confidence': confidence})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)