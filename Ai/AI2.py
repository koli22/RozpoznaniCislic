import tensorflow as tf
import numpy as np
import os
from PIL import Image
from tensorflow.keras.utils import to_categorical

def load_data(data_dir, img_size):
    images = []
    labels = []
    for label in range(10):  # For labels 0 to 9
        label_dir = os.path.join(data_dir, f"{label}o")
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                try:
                    img = Image.open(img_path).convert('L')  # Convert to grayscale
                    img = img.resize((img_size, img_size))
                    img_array = np.array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels)

# Parameters
test_data_dir = 'test_data1'  # Path to your test dataset (change to your test dataset directory)
img_size = 25  # Image size (25x25)
num_classes = 10  # Number of classes (0-9)

# Load the saved model
model = tf.keras.models.load_model('digit_recognition_model.h5')
print("Model loaded from digit_recognition_model.h5")

# Verify the model architecture
model.summary()

# Load and preprocess test data
X_test, y_test = load_data(test_data_dir, img_size)
print(f"Loaded {len(X_test)} test images with {len(y_test)} labels.")
if len(X_test) == 0 or len(y_test) == 0:
    raise ValueError("No test data loaded. Please check the test data directory and files.")
X_test = X_test.reshape(-1, img_size, img_size, 1) / 255.0
y_test = to_categorical(y_test, num_classes)

# Evaluate the model on test data
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")
