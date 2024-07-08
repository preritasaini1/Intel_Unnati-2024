import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, PReLU

# Function definitions for pixelation detection and SRCNN model

def preprocess_image(image):
    new_dim = (300, 280)
    resized_image = image.resize(new_dim)
    return resized_image

def detect_edges(image):
    gray = image.convert('L')
    edges = cv2.Canny(np.array(gray), 100, 200)
    return edges

def train_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def extract_features(edges):
    vertical_lines = np.sum(np.abs(np.diff(edges, axis=0)), axis=0)
    horizontal_lines = np.sum(np.abs(np.diff(edges, axis=1)), axis=1)
    features = np.concatenate((vertical_lines, horizontal_lines))
    return features

def predict_pixelation(model, image):
    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image)
    features = extract_features(edges).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

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

# Directory where dataset is stored
dataset_dir = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024//train'

# Initialize empty lists for images and labels
images = []
labels = []

# Iterate through each image file in the directory
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg') or filename.endswith('.jpeg') or filename.endswith('.png'):
        img_path = os.path.join(dataset_dir, filename)
        images.append(img_path)
        if '1' in filename.lower():
            labels.append(1)  # Assign label 1 for pixelated images
        else:
            labels.append(0)  # Assign label 0 for non-pixelated images

# Prepare the dataset
X = []
y = labels
for img_path in images:
    img = Image.open(img_path)
    preprocessed_image = preprocess_image(img)
    edges = detect_edges(preprocessed_image)
    features = extract_features(edges)
    X.append(features)

X = np.array(X)
y = np.array(y)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the logistic regression model
model = train_model(X_train, y_train)

# Evaluate the logistic regression model
y_pred = model.predict(X_test)
f1 = f1_score(y_test, y_pred)
precision, recall, _ = precision_recall_curve(y_test, y_pred)

print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# Predict pixelation in a new image
new_image_path = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024//train//Scen1.jpg'
new_image = Image.open(new_image_path)
is_pixelated = predict_pixelation(model, new_image)
print(f"Is the image pixelated? {'Yes' if is_pixelated else 'No'}")


# Apply SRCNN model if image is pixelated
if is_pixelated:
    # Directory paths
    HR_folder = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024//high_pic'
    LR_folder = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024//low_pic'

    # Load and preprocess HR and LR images
    HR_images = load_images_from_folder(HR_folder, size=(128, 128))
    LR_images = load_images_from_folder(LR_folder, size=(128, 128))

    HR_images = preprocess_images(HR_images)
    LR_images = preprocess_images(LR_images)

    # Build and train SRCNN model
    srcnn_model = build_complex_SRCNN()
    srcnn_model.fit(LR_images, HR_images, epochs=300, batch_size=16)

    # Test SRCNN model on new image
    test_img = cv2.imread(new_image_path)
    original_size = (test_img.shape[1], test_img.shape[0])
    test_img_resized = cv2.resize(test_img, (128, 128))
    test_img_normalized = test_img_resized.astype('float32') / 255.0
    test_img_normalized = np.expand_dims(test_img_normalized, axis=0)

    result = srcnn_model.predict(test_img_normalized)
    result = result.squeeze()
    result = (result * 255.0).astype('uint8')
    result = cv2.resize(result, original_size)

    # Save and display the result
    output_path = 'C://Users//Lenovo//Desktop//Intel_Unnati 2024/result.jpg'
    cv2.imwrite(output_path, result)

    # Display the images
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

else:
    # Display original image if not pixelated
    plt.imshow(new_image)
    plt.title('Original Image')
    plt.axis('off')
    plt.show()
