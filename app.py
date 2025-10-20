
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os
import io
from tflite_runtime.interpreter import Interpreter

IMG_SIZE = 224
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB limit



# Load TFLite model and classes using LiteRT
interpreter = Interpreter(model_path='plant_classifier_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
with open('class_names.txt') as f:
    class_names = f.read().splitlines()

app = Flask(__name__)
CORS(app)  # Enable CORS for React Native requests
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint for Render"""
    return jsonify({'status': 'healthy', 'interpreter_loaded': interpreter is not None}), 200

@app.route('/predict', methods=['POST'])
def predict():
    # Accept up to 4 images under the same key: images[]
    files = request.files.getlist('images')
    if not files or len(files) == 0:
        return jsonify({'error': 'No images provided. Please upload 1-4 images using the key images[].'}), 400
    if len(files) > 4:
        return jsonify({'error': 'Too many images. Maximum is 4.'}), 400
    results = []
    for idx, file in enumerate(files):
        # Check file extension
        if not file.filename or not (file.filename.lower().endswith(('.jpg', '.jpeg', '.png'))):
            results.append({'error': f'Unsupported file type for {file.filename or "unknown"}', 'index': idx})
            continue
        
        # Check if file is empty
        file.seek(0, os.SEEK_END)
        file_size = file.tell()
        file.seek(0)
        
        if file_size == 0:
            results.append({'error': 'Empty file', 'filename': file.filename, 'index': idx})
            continue
        
        if file_size > MAX_FILE_SIZE:
            results.append({'error': f'File too large (max {MAX_FILE_SIZE // (1024*1024)}MB)', 'filename': file.filename, 'index': idx})
            continue
            
        try:
            # Read file into BytesIO for compatibility with gunicorn
            file_bytes = io.BytesIO(file.read())
            img = keras.preprocessing.image.load_img(file_bytes, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            # If you normalized during training, normalize here as well
            # img_array = img_array / 255.0

            # Set input tensor
            interpreter.set_tensor(input_details[0]['index'], img_array.astype(input_details[0]['dtype']))
            interpreter.invoke()
            predictions = interpreter.get_tensor(output_details[0]['index'])[0]
            predicted_class = class_names[np.argmax(predictions)]
            confidence = float(np.max(predictions))

            # Return top 3 predictions for better insights
            top_3_idx = np.argsort(predictions)[-3:][::-1]
            top_3_predictions = [
                {'class': class_names[i], 'confidence': float(predictions[i])}
                for i in top_3_idx
            ]

            results.append({
                'class': predicted_class,
                'confidence': confidence,
                'filename': file.filename,
                'index': idx,
                'top_predictions': top_3_predictions
            })
        except Exception as e:
            app.logger.error(f'Error processing {file.filename}: {str(e)}')
            results.append({'error': str(e), 'filename': file.filename, 'index': idx})
    
    return jsonify({'results': results, 'total_images': len(files)})

@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file size exceeded"""
    return jsonify({'error': f'File too large. Maximum size is {MAX_FILE_SIZE // (1024*1024)}MB'}), 413

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)