import os
import cv2
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve

def preprocess_image(image):
    new_dim = (300, 280)
    resized_image = image.resize(new_dim)
    return resized_image

def detect_edges(image):
    gray = image.convert('L')
    # Detect edges using Canny (not available in PIL, so we use OpenCV)
    edges = cv2.Canny(np.array(gray), 100, 200)
    return edges

def train_model(X, y):
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    return model

def extract_features(edges):
    # Count the number of vertical and horizontal lines
    vertical_lines = np.sum(np.abs(np.diff(edges, axis=0)), axis=0)
    horizontal_lines = np.sum(np.abs(np.diff(edges, axis=1)), axis=1)
    # Aggregate features
    features = np.concatenate((vertical_lines, horizontal_lines))
    return features

def predict_pixelation(model, image):
    preprocessed_image = preprocess_image(image)
    edges = detect_edges(preprocessed_image)
    features = extract_features(edges).reshape(1, -1)
    prediction = model.predict(features)
    return prediction[0]

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

# Train the model
model = train_model(X_train, y_train)

# Evaluate the model
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
