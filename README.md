# Face Mask Detection System 😷

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A robust, real-time Computer Vision system designed to detect human faces and verify mask compliance. This project utilizes a **Hybrid Architecture** combining the speed of **Haar Cascade Classifiers** for detection with the accuracy of a **Deep Learning (CNN)** model for classification.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
    - [1. Face Localization](#1-face-localization-viola-jones)
    - [2. Mask Classification](#2-mask-classification)
- [Tech Stack Details](#-tech-stack-details)
- [Installation Guide](#-installation-guide)
- [Usage Instructions](#-usage-instructions)
- [Dataset & Model Info](#-dataset--model-info)
- [Troubleshooting](#-troubleshooting)
- [Future Enhancements](#-future-enhancements)
- [License](#-license)

---

## 🚀 Project Overview

The COVID-19 pandemic highlighted the importance of automated public health monitoring. This system provides a non-intrusive way to check for face masks in real-time video streams.

**Why this approach?**
Detecting objects (faces) and classifying them are two different problems.
*   **Detection**: Needs to be incredibly fast to scan every pixel of a video frame. We use **Haar Cascades** (Traditional CV) because they are lightweight and extremely efficient on CPUs.
*   **Classification**: Needs to look at features (texture, edges) to distinguish "Mask" vs "Mouth/Nose". We use a trained **Convolutional Neural Network (CNN)** because it captures complex patterns effectively.

---

## ✨ Key Features

*   **Real-Time Performance**: Achieves smooth frame rates on standard hardware.
*   **Low Latency**: The hybrid approach minimizes processing time per frame.
*   **Visual Feedback**: Color-coded bounding boxes (Green/Red) and labels on the live video stream.
*   **Robustness**: Handles various lighting conditions reasonably well.
*   **Educational Value**: The included Jupyter Notebook (`dL-mask.ipynb`) serves as a step-by-step tutorial on building the pipeline.

---

## 🏗 System Architecture

### 1. Face Localization (Viola-Jones)
*   **Algorithm**: Haar Feature-based Cascade Classifier.
*   **File**: `haarcascade_frontalface_alt2.xml`
*   **Methodology**:
    *   Uses **Integral Images** to compute rectangular features in $O(1)$ time.
    *   Uses **Attentional Cascade** structure: Simple classifiers run first (e.g., checking for eye regions). If they fail, the window is discarded immediately. Only potential faces trigger complex processing.

### 2. Mask Classification (Deep Learning)
*   **Model**: Keras/TensorFlow `.h5` model.
*   **Architecture**:
    *   A custom trained **Convolutional Neural Network (CNN)** designed for binary classification (Mask / No Mask).
    *   It takes the cropped face image as input and outputs the probability of a mask being present.

---

## 🛠 Tech Stack Details

| Component | Library | Version | Description |
| :--- | :--- | :--- | :--- |
| **Language** | Python | 3.8+ | Primary development language. |
| **CV Library** | OpenCV (`cv2`) | 4.5+ | Handling video streams and Haar Cascades. |
| **ML Backend** | TensorFlow | 2.5+ | Running the Keras model inference. |
| **Math** | NumPy | 1.19+ | Tensor manipulation and image pre-processing. |
| **Environment** | Jupyter | - | Rapid prototyping and documentation. |

---

## 📦 Installation Guide

It is highly recommended to use a virtual environment to avoid conflicts.

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/vargheesk/Mask-detection-Hybrid.git
    cd Mask-detection-Hybrid
    ```

2.  **Create Virtual Environment**
    ```bash
    # Windows
    python -m venv venv
    .\venv\Scripts\activate

    # Mac/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Dependencies**
    ```bash
    pip install opencv-python tensorflow numpy jupyter notebook
    ```

---

## ⏯ Usage Instructions

1.  **Launch Jupyter Notebook**:
    ```bash
    jupyter notebook dL-mask.ipynb
    ```
2.  **Execute Cells**: Run the cells sequentially. The notebook is structured to:
    *   Explain the theory.
    *   Load the models.
    *   Test on a static image (`masked.jpeg`).
    *   **Final Cell**: Starts the webcam loop.
3.  **Stop**: Press **`q`** in the video window to terminate the application.

---

## 📊 Dataset & Model Info

*   **Input Shape**: `(224, 224, 3)`
*   **Preprocessing**: Pixel scaling `[-1, 1]`.
*   **Training**: The model `mask_recog.h5` was trained on a mixed dataset of masked and unmasked face images to learn the distinguishing features of surgical and cloth masks.

---

## ❓ Troubleshooting

**Q: The camera doesn't open?**
*   **A**: Ensure no other app (Zoom, Teams) is using the camera. Check if `cv2.VideoCapture(0)` index is correct (try 1 or 2 if you have external cams).

**Q: `AttributeError: module 'cv2' has no attribute 'face'`?**
*   **A**: You might need `opencv-contrib-python`. Run `pip install opencv-contrib-python`.

**Q: Low FPS on my machine?**
*   **A**: The Haar Cascade `minNeighbors` or `scaleFactor` can be tuned. High resolution video feeds also slow down processing; try resizing the frame in the code: `frame = cv2.resize(frame, (640, 480))`.

---

## 🔮 Future Enhancements

- [ ] **Face Recognition**: Identify *who* is behind the mask.
- [ ] **Incorrect Mask Detection**: Classify "Nose Visible" or "Chin Visible" as improper usage.
- [ ] **Multi-Camera Support**: Deploy on CCTV feeds via RTSP.
- [ ] **Web Dashboard**: Create a Flask/Streamlit frontend for analytics.

---

## 📜 License

This project is licensed under the **MIT License**. Feel free to fork, modify, and distribute.

---
*Created by [Vargheesk](https://github.com/vargheesk)*
