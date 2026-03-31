# Speech-Emotion-Analyzer
A machine learning-powered web application that detects human emotions from speech in real-time. Built with Python, Scikit-Learn, and Librosa, and deployed via Streamlit.

---

## 🚀 Overview
This application allows users to upload `.wav` audio files and receive an instant prediction of the emotion expressed. It is particularly useful for sentiment analysis, human-computer interaction research, and accessibility tools.

### ✨ Key Features
* **Instant Analysis:** Upload and process audio files in seconds.
* **Feature Extraction:** Extracts MFCCs (40 features), Chroma, and Mel-spectrograms using `Librosa`.
* **User-Friendly UI:** A clean, modern interface built with **Streamlit**.
* **Live Playback:** Listen to the uploaded audio directly within the dashboard.

---

## 📊 Dataset
The model is trained on the **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.
* **Actors:** 24 professional actors (12 female, 12 male).
* **Emotions Included:** Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, and Surprised.
* **Naming Convention:** Files follow a specific numerical ID (e.g., `03-01-05...` where `05` represents "Angry").

---

## 🛠️ Tech Stack
* **Language:** Python
* **Web Framework:** Streamlit
* **Audio Processing:** Librosa, Soundfile
* **Machine Learning:** Scikit-Learn, Pickle
* **Data Manipulation:** Numpy

---
# 🎥 Project Demo
This video demonstrates the working of the Speech Emotion Recognition system.
🔗 Click below to watch:
https://drive.google.com/drive/folders/1gwGvAg3RSS0--XQAb4bDfv0pTXgsr06R?usp=sharing

## 📂 Project Structure
```text
├── app.py                     # Main Streamlit application code
├── Algonive_Emotion_Model.pkl # Pre-trained ML model (Pickle format)
├── requirements.txt           # Dependencies for deployment
└── README.md                  # Project documentation
