import streamlit as st
import librosa
import numpy as np
import pickle
import soundfile
import io

# --- 1. FEATURE EXTRACTION FUNCTION ---
def extract_feature(file_name, mfcc, chroma, mel):
    # Load the audio file from the Streamlit UploadedFile object
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T, axis=0)
            result = np.hstack((result, mel))
    return result

# --- 2. THE APP ---
st.title("🎙️ Speech Emotion Analyzer")

# Load your specific model
# Ensure this .pkl file is in the same folder as your notebook
model = pickle.load(open("Algonive_Emotion_Model.pkl", "rb"))

uploaded_file = st.file_uploader("Upload Audio", type="wav")

if uploaded_file:
    st.audio(uploaded_file)
    
    # Extract features matching your model's training (MFCC, Chroma, Mel)
    features = extract_feature(uploaded_file, mfcc=True, chroma=True, mel=True)
    
    # Predict using the loaded model
    prediction = model.predict(features.reshape(1, -1))
    
    st.success("Analysis Complete!")
    st.header(f"Predicted Emotion: {prediction[0]}")
