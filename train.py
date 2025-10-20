import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from PIL import Image

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
DATA_DIR = 'dataset'  # ← UPDATE THIS to your dataset folder path

# ========== ADD THIS: Clean invalid images ==========
def clean_dataset(data_dir):
    """Remove invalid/corrupted images from dataset using TensorFlow validation"""
    valid_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
    removed_count = 0
    
    for root, dirs, files in os.walk(data_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            file_ext = os.path.splitext(filename)[1].lower()
            
            # Check file extension
            if file_ext not in valid_extensions:
                print(f"Removing non-image file: {filepath}")
                try:
                    os.remove(filepath)
                    removed_count += 1
                except Exception as e:
                    print(f"  Could not remove: {e}")
                continue
            
            # Verify with PIL
            try:
                with Image.open(filepath) as img:
                    img.verify()
                with Image.open(filepath) as img:
                    img.load()
            except Exception as e:
                print(f"Removing corrupted image (PIL): {filepath} - Error: {e}")
                try:
                    os.remove(filepath)
                    removed_count += 1
                except:
                    pass
                continue
            
            # Verify with TensorFlow (more strict - catches what PIL misses)
            try:
                img_data = tf.io.read_file(filepath)
                img = tf.image.decode_image(img_data, channels=3, expand_animations=False)
                # Try to actually use the image
                img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
                img.numpy()  # Force evaluation
            except Exception as e:
                print(f"Removing corrupted image (TF): {filepath} - Error: {e}")
                try:
                    os.remove(filepath)
                    removed_count += 1
                except:
                    pass
                continue
    
    print(f"\nCleaning complete! Removed {removed_count} invalid files.")
    return removed_count

# Clean the dataset before training
print("Checking dataset for invalid images...")
clean_dataset(DATA_DIR)
print("\n" + "="*50 + "\n")

# ========== Rest of your code continues here ==========

# 1. Load and prepare data
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Get class names
class_names = train_ds.class_names
print(f"Found {len(class_names)} plant species: {class_names}")

# 2. Optimize performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 3. Data augmentation (improves model robustness)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 4. Build model using transfer learning (recommended for plant classification)
def create_model(num_classes):
    # Use a pre-trained model as base
    base_model = keras.applications.MobileNetV2(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model
    base_model.trainable = False
    
    # Build model
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = data_augmentation(inputs)
    x = keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs)
    return model

# Create and compile model
model = create_model(len(class_names))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train model
print("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2
        )
    ]
)


# 6. Save model
model.save('plant_classifier_model.keras')
print("Model saved!")

# 6b. Convert and save as TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('plant_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)
print("TensorFlow Lite model saved!")

# Save class names for later use
with open('class_names.txt', 'w') as f:
    f.write('\n'.join(class_names))

# 7. Evaluate model
loss, accuracy = model.evaluate(val_ds)
print(f"\nValidation accuracy: {accuracy*100:.2f}%")

# 8. Make predictions (example)
def predict_plant(image_path, model, class_names):
    img = keras.preprocessing.image.load_img(
        image_path,
        target_size=(IMG_SIZE, IMG_SIZE)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    return predicted_class, confidence

# Example usage:
# predicted_species, confidence = predict_plant('test_image.jpg', model, class_names)
# print(f"Predicted: {predicted_species} (confidence: {confidence*100:.2f}%)")