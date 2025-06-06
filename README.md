# Smart Image Sorter with Tray App (TensorFlow + scikit-learn)

This is a desktop automation utility built using **TensorFlow**, **scikit-learn**, and **PyStray**. It monitors a folder for incoming images, classifies them using a pre-trained machine learning model, and automatically moves them into sorted folders based on the predicted class.

The app includes a tray icon for quick access to settings and model retraining.

---

## Features

- Monitors a folder in real time for new image files
- Uses deep learning (MobileNetV2) for feature extraction
- Uses a machine learning classifier for predicting image class
- Automatically moves files to folders named after predicted labels
- Displays desktop notifications with prediction confidence
- Provides system tray interface with:
  - Open Settings
  - Force Retrain
  - Exit

---

## UI Overview

- **System Tray Menu**: Right-click to access:
  - Settings GUI
  - Force model retraining
  - Exit application
- **Desktop Notifications**:
  - Shows predicted class and confidence for each image processed
- **Folder Watcher**:
  - Watches for `.jpg`, `.jpeg`, `.png`, `.webp` files
  - Handles unknown and non-image files separately

---

## Requirements

- Python 3.8+
- `tensorflow`
- `numpy`
- `pillow`
- `pystray`
- `joblib`
- `shutil`
- `plyer`
- `scikit-learn`

### Install dependencies:

```bash
pip install tensorflow numpy pillow pystray joblib plyer scikit-learn
