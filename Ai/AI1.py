import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import os
from PIL import Image

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
data_dir = 'data'  # Path to your dataset (change to your dataset directory)
img_size = 25  # Image size (25x25)
num_classes = 10  # Number of classes (0-9)

# Load and preprocess data
X, y = load_data(data_dir, img_size)
print(f"Loaded {len(X)} images with {len(y)} labels.")
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Please check the data directory and files.")
X = X.reshape(-1, img_size, img_size, 1) / 255.0  # Normalize and reshape
y = to_categorical(y, num_classes)  # One-hot encode labels

# Build the neural network
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

model.save('digit_recognition_model.h5')


# Assuming you have a separate test dataset
def load_test_data(data_dir, img_size):
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

# Load and preprocess test data
test_data_dir = 'test_data'  # Path to your test dataset (change to your test dataset directory)
X_test, y_test = load_test_data(test_data_dir, img_size)
print(f"Loaded {len(X_test)} test images with {len(y_test)} labels.")
if len(X_test) == 0 or len(y_test) == 0:
    raise ValueError("No test data loaded. Please check the test data directory and files.")
X_test = X_test.reshape(-1, img_size, img_size, 1) / 255.0
y_test = to_categorical(y_test, num_classes)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

