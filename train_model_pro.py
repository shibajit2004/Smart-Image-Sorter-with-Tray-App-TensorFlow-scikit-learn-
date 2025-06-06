import os
import random
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load settings
with open("settings.json", "r") as f:
    settings = json.load(f)

PHOTO_DB_FOLDER = settings["PHOTO_DB_FOLDER"]
MODEL_PATH = "D:\\Shibajit Chatterjee\\Study\\Third Year\\6th Semister\\Machine learning\\LAB\\ImageFileOrgaizer\\rf_model.pkl"
ENCODER_PATH = "D:\\Shibajit Chatterjee\\Study\\Third Year\\6th Semister\\Machine learning\\LAB\\ImageFileOrgaizer\\label_encoder.pkl"
FEATURE_EXTRACTOR_PATH = "D:\\Shibajit Chatterjee\\Study\\Third Year\\6th Semister\\Machine learning\\LAB\ImageFileOrgaizer\\mobilenetv2_feature_extractor.keras" 

# Load saved MobileNetV2 feature extractor
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)

IMG_SIZE = (128, 128)

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

# Determine minimum number of images across valid folders (folders with >= 20 images)
folder_image_counts = []

for label in os.listdir(PHOTO_DB_FOLDER):
    label_path = os.path.join(PHOTO_DB_FOLDER, label)
    if os.path.isdir(label_path):
        images = [img for img in os.listdir(label_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= 20:
            folder_image_counts.append(len(images))

if not folder_image_counts:
    raise Exception("No valid folders found with at least 20 images.")

min_images = min(folder_image_counts)
print(f"Minimum images across valid folders: {min_images}")

if min_images < 20:
    print("Warning: Classes have very few images. Consider adding more.")

# Extract features and labels
X, y = [], []

for label in os.listdir(PHOTO_DB_FOLDER):
    label_path = os.path.join(PHOTO_DB_FOLDER, label)
    if os.path.isdir(label_path):
        images = [img for img in os.listdir(label_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= 20:
            random.shuffle(images)
            selected_images = images[:min_images]
            for img_file in selected_images:
                img_path = os.path.join(label_path, img_file)
                try:
                    features = extract_features(img_path)
                    X.append(features)
                    y.append(label)
                except Exception as e:
                    print(f"Skipping {img_file}: {e}")
        else:
            print(f"Skipping folder '{label}' because it has only {len(images)} images (<20).")

X = np.array(X)
y = np.array(y)

print(f"Final dataset size: {X.shape[0]} samples.")

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# Adjust test size if dataset is very small
test_size = 0.2 if len(y_encoded) >= 10 else 0.5

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
)

# Train RandomForest model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate model
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("\nTraining Complete")
print(f"Accuracy: {acc:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and encoder
joblib.dump(clf, MODEL_PATH)
joblib.dump(encoder, ENCODER_PATH)

print("\nModel and encoder saved successfully.")
