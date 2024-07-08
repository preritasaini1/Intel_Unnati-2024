import tensorflow as tf
import cv2
import numpy as np
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, PReLU

# Load and Preprocess Data
def load_images_from_folder(folder, size=(128, 128)):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            img = cv2.resize(img, size)
            images.append(img)
    return images

def preprocess_images(images):
    processed_images = []
    for img in images:
        img = img.astype('float32') / 255.0
        processed_images.append(img)
    return np.array(processed_images)

HR_folder = 'C://Users//Lenovo//Desktop//Practice_intel//high_pic'  
LR_folder = 'C://Users//Lenovo//Desktop//Practice_intel//low_pic'  

# Ensure HR and LR images are resized to the same dimensions
HR_images = load_images_from_folder(HR_folder, size=(128, 128))
LR_images = load_images_from_folder(LR_folder, size=(128, 128))

HR_images = preprocess_images(HR_images)
LR_images = preprocess_images(LR_images)

# Build a More Complex SRCNN Model
def build_complex_SRCNN():
    model = Sequential()
    model.add(Conv2D(64, (9, 9), padding='same', input_shape=(128, 128, 3)))
    model.add(PReLU())
    model.add(Conv2D(32, (1, 1), padding='same'))
    model.add(PReLU())
    model.add(Conv2D(3, (5, 5), padding='same'))
    model.add(PReLU())
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = build_complex_SRCNN()

# Train the Model
model.fit(LR_images, HR_images, epochs=300, batch_size=16)  

# Test the Model
test_img_path = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024//low_pic/Scen1.jpg'  
test_img = cv2.imread(test_img_path)
original_size = (test_img.shape[1], test_img.shape[0])  
test_img_resized = cv2.resize(test_img, (128, 128))
test_img_normalized = test_img_resized.astype('float32') / 255.0
test_img_normalized = np.expand_dims(test_img_normalized, axis=0)

result = model.predict(test_img_normalized)
result = result.squeeze()
result = (result * 255.0).astype('uint8')
result = cv2.resize(result, original_size)  

# Save and Display the Result
output_path = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024/result.jpg'  

# Display the images
import matplotlib.pyplot as plt

# Display the images using Matplotlib
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB))
plt.title('Input Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
plt.title('Super-Resolved Image')
plt.axis('off')

plt.tight_layout()
plt.show()

