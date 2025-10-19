"""
Plant identification API - receives images and returns predictions
Can handle single or multiple images
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import json
from pathlib import Path

class PlantIdentifier:
    def __init__(self, model_path='plant_classifier_model.keras', class_names_path='class_names.txt'):
        """Initialize the plant identifier"""
        self.model = keras.models.load_model(model_path)
        
        with open(class_names_path, 'r') as f:
            self.class_names = [line.strip() for line in f.readlines()]
        
        print(f"✓ Model loaded with {len(self.class_names)} plant species", file=sys.stderr)
    
    def preprocess_image(self, image_path):
        """Preprocess a single image"""
        img = keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
        return img_array
    
    def identify_single_image(self, image_path):
        """Identify plant from single image"""
        if not Path(image_path).exists():
            return {"error": f"Image not found: {image_path}"}
        
        try:
            # Preprocess and predict
            img_array = self.preprocess_image(image_path)
            predictions = self.model.predict(img_array, verbose=0)
            
            # Get top 3 predictions
            top_3_indices = np.argsort(predictions[0])[-3:][::-1]
            
            results = []
            for idx in top_3_indices:
                results.append({
                    "identifier": self.class_names[idx],
                    "confidence": float(predictions[0][idx]),
                    "confidence_percentage": round(float(predictions[0][idx]) * 100, 2)
                })
            
            return {
                "success": True,
                "image_path": image_path,
                "predictions": results
            }
        
        except Exception as e:
            return {"error": f"Failed to process {image_path}: {str(e)}"}
    
    def identify_multiple_images(self, image_paths):
        """Identify plants from multiple images and combine results"""
        if not image_paths:
            return {"error": "No image paths provided"}
        
        all_results = []
        combined_predictions = {}
        
        # Process each image
        for image_path in image_paths:
            result = self.identify_single_image(image_path)
            all_results.append(result)
            
            if result.get("success"):
                # Combine predictions (weighted average)
                for pred in result["predictions"]:
                    identifier = pred["identifier"]
                    confidence = pred["confidence"]
                    
                    if identifier in combined_predictions:
                        combined_predictions[identifier].append(confidence)
                    else:
                        combined_predictions[identifier] = [confidence]
        
        # Calculate final combined results
        final_predictions = []
        for identifier, confidences in combined_predictions.items():
            avg_confidence = np.mean(confidences)
            final_predictions.append({
                "identifier": identifier,
                "confidence": float(avg_confidence),
                "confidence_percentage": round(float(avg_confidence) * 100, 2),
                "appears_in_images": len(confidences)
            })
        
        # Sort by confidence
        final_predictions.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "num_images": len(image_paths),
            "individual_results": all_results,
            "combined_predictions": final_predictions[:3]  # Top 3
        }

def main():
    """Main function for command line usage"""
    # Initialize identifier
    identifier = PlantIdentifier()
    
    # Get image paths from command line, or use default test image
    if len(sys.argv) < 2:
        # Use default test image
        image_paths = ['DSC_0238.jpg']
        print("No image specified, using DSC_0238.jpg", file=sys.stderr)
    else:
        image_paths = sys.argv[1:]
    
    if len(image_paths) == 1:
        # Single image
        result = identifier.identify_single_image(image_paths[0])
    else:
        # Multiple images
        result = identifier.identify_multiple_images(image_paths)
    
    # Output JSON result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()