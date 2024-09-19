import streamlit as st
import numpy as np
import librosa
from keras.models import load_model
import joblib

# Load pre-trained model and OneHotEncoder
model = load_model('emotion_detection_model.h5')
enc = joblib.load('onehotencoder.pkl')

st.title('🎤 Emotion Detection from Audio')

uploaded_file = st.file_uploader(
    "Upload an audio file in WAV format to detect the emotion.",
    type="wav"
)

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    # Save uploaded file temporarily
    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())
    
    def extract_mfcc(filename):
        y, sr = librosa.load(filename, duration=3, offset=0.5)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc

    with st.spinner('Processing audio and predicting emotion...'):
        mfcc = extract_mfcc("temp_audio.wav")
        mfcc = np.expand_dims(mfcc, -1)
        mfcc = np.expand_dims(mfcc, 0)

        # Predict emotion
        prediction = model.predict(mfcc)
        emotion = enc.inverse_transform(prediction)
        
        emotion_str = str(emotion[0])
        emotion_str = emotion_str.strip("[]'")
        emotion_str = emotion_str.capitalize()
        
        st.success(f"The detected emotion is: **{emotion_str}**")
        
        st.balloons()
