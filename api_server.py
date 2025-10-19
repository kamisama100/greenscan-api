"""
Flask API server for plant identification
Mobile app sends images via HTTP POST, receives JSON predictions
"""

from flask import Flask, request, jsonify
import os
import uuid
from werkzeug.utils import secure_filename
from identify import PlantIdentifier
import json

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = 'temp_uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the plant identifier once when server starts
print("Loading AI model...")
identifier = PlantIdentifier()
print("✅ API Server ready!")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/identify', methods=['POST'])
def identify_plant():
    """
    API endpoint to identify plants from uploaded images
    
    Expected: multipart/form-data with 1-4 image files
    Returns: JSON with predictions
    """
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        
        if not files or len(files) == 0:
            return jsonify({'error': 'No images selected'}), 400
        
        if len(files) > 4:
            return jsonify({'error': 'Maximum 4 images allowed'}), 400
        
        # Save uploaded files temporarily
        temp_paths = []
        for file in files:
            if file and allowed_file(file.filename):
                # Generate unique filename
                filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                temp_paths.append(filepath)
            else:
                return jsonify({'error': f'Invalid file type: {file.filename}'}), 400
        
        # Identify plant(s)
        if len(temp_paths) == 1:
            result = identifier.identify_single_image(temp_paths[0])
        else:
            result = identifier.identify_multiple_images(temp_paths)
        
        # Clean up temporary files
        for path in temp_paths:
            try:
                os.remove(path)
            except:
                pass  # Ignore cleanup errors
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_classes': len(identifier.class_names),
        'available_species': identifier.class_names
    })

@app.route('/', methods=['GET'])
def home():
    """API documentation"""
    return jsonify({
        'message': 'Plant Identification API',
        'endpoints': {
            'POST /identify': 'Upload 1-4 images for plant identification',
            'GET /health': 'Check API status',
            'GET /': 'This documentation'
        },
        'usage': {
            'method': 'POST',
            'url': '/identify',
            'content_type': 'multipart/form-data',
            'field_name': 'images',
            'max_files': 4,
            'max_size': '16MB per file',
            'formats': ['jpg', 'jpeg', 'png', 'gif']
        }
    })

if __name__ == '__main__':
    # Run the development server
    app.run(host='0.0.0.0', port=5000, debug=True)