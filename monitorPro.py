import os
import time
import shutil
import subprocess
import json
import threading
import pystray
from pystray import MenuItem as item
from PIL import Image
import numpy as np
import tensorflow as tf
import joblib
from plyer import notification

# ========== Load Settings ==========
SETTINGS_FILE = "settingsNew.json"

def load_settings():
    with open(SETTINGS_FILE, "r") as f:
        return json.load(f)

settings = load_settings()

WATCH_FOLDER = settings["WATCH_FOLDER"]
PHOTO_DB_FOLDER = settings["PHOTO_DB_FOLDER"]  # Used only for training
SORTED_FOLDER = settings.get("SORTED_FOLDER", "Sorted Output")  # Folder to move predicted images
MODEL_PATH = settings["MODEL_PATH"]
ENCODER_PATH = settings["ENCODER_PATH"]
FEATURE_EXTRACTOR_PATH = settings["FEATURE_EXTRACTOR_PATH"]

clf = joblib.load(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

IMG_SIZE = (128, 128)
feature_extractor = tf.keras.models.load_model(FEATURE_EXTRACTOR_PATH)

MIN_CONFIDENCE = 0.35
running = True

NOTIFY_ICON_PATH = "assets\\icon.ico"

def extract_features(img_path):
    img = Image.open(img_path).convert('RGB').resize(IMG_SIZE)
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    features = feature_extractor.predict(img_array, verbose=0)
    return features.flatten()

def resolve_filename_conflict(dest_folder, filename):
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    while os.path.exists(os.path.join(dest_folder, new_filename)):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1
    return new_filename

def predict_and_move(image_path):
    try:
        features = extract_features(image_path).reshape(1, -1)
        probs = clf.predict_proba(features)[0]
        best_idx = np.argmax(probs)
        confidence = probs[best_idx]

        predicted_label = encoder.inverse_transform([best_idx])[0] if confidence >= MIN_CONFIDENCE else "unknown"

        # Destination: inside SORTED_FOLDER
        dest_folder = os.path.join(SORTED_FOLDER, predicted_label)
        os.makedirs(dest_folder, exist_ok=True)

        filename = resolve_filename_conflict(dest_folder, os.path.basename(image_path))
        destination = os.path.join(dest_folder, filename)
        shutil.move(image_path, destination)

        notification.notify(
            title='Image Sorted',
            message=f"'{os.path.basename(image_path)}' -> '{predicted_label}' ({confidence:.2%})",
            app_icon=NOTIFY_ICON_PATH,
            timeout=10
        )

        print(f"Moved {image_path} -> {destination} (Predicted: {predicted_label}, Confidence: {confidence:.2%})")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")

def background_monitor():
    print(f"Monitoring '{WATCH_FOLDER}' for new files...")
    while running:
        for file in os.listdir(WATCH_FOLDER):
            file_path = os.path.join(WATCH_FOLDER, file)
            if not os.path.isfile(file_path):
                continue
            ext = os.path.splitext(file)[-1].lower()
            if ext in ('.png', '.jpg', '.jpeg','.webp'):
                predict_and_move(file_path)
            else:
                file_type_folder = ext[1:].upper()
                os.makedirs(file_type_folder, exist_ok=True)
                new_filename = resolve_filename_conflict(file_type_folder, file)
                dest_path = os.path.join(file_type_folder, new_filename)
                try:
                    shutil.move(file_path, dest_path)
                    print(f"Moved non-image file {file} -> {file_type_folder}")
                except Exception as e:
                    print(f"Error moving {file_path}: {e}")
        time.sleep(2)

def open_settings_gui(icon, item):
    subprocess.Popen(["python", "settings_guiNew.py"])

def force_retrain(icon, item):
    subprocess.Popen(["python", "train_model_pro.py"])
    notification.notify(
        title='Model Retraining',
        message='Retraining process triggered manually.',
        app_icon=NOTIFY_ICON_PATH,
        timeout=10
    )

def on_exit(icon, item):
    global running
    running = False
    icon.stop()

def setup_tray():
    icon_image = Image.open("assets\\icon.png")
    menu = pystray.Menu(
        item('Force Retrain', force_retrain),
        item('Settings', open_settings_gui),
        item('Exit', on_exit)
    )
    tray_icon = pystray.Icon("ImageMonitor", icon=icon_image, menu=menu)
    tray_icon.run()

if __name__ == "__main__":
    threading.Thread(target=background_monitor, daemon=True).start()
    setup_tray()
