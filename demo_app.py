import streamlit as st
import whisper
import nltk
import tempfile
from nltk.sentiment import SentimentIntensityAnalyzer
import librosa
import numpy as np
import tensorflow as tf
import os
import soundfile as sf
import matplotlib.pyplot as plt

# Title and Description
st.title("Voice Sentiment Analysis")
st.markdown("""
This app is the final project for DSBA 6156: Applied Machine Learning, Fall 2024.  
__Team: Eric Phann & Jessica Ricks.__  
            
This app combines __three approaches__ to Voice Sentiment Analysis:  
  1. Transcribing audio to text using OpenAI's Whisper model
  2. Classifying the _text_ as negative, neutral, or positive using NLTK's Sentiment Intensity Analysis model
  3. Classifying the _audio_ as negative, neutral, or positive using our pre-trained convolutional neural network (CNN). 
            The inputs are Mel-Frequency Cepstral Coefficients (MFCC) extracted from the audio.
""")

# Initialize Whisper and Sentiment Analyzer
@st.cache_resource
def load_whisper_model():
    return whisper.load_model("base.en") # base multilingual model

@st.cache_resource
def load_sentiment_analyzer():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

whisper_model = load_whisper_model()
sentiment_analyzer = load_sentiment_analyzer()

# Function to extract MFCCs
def extract_mfcc(file_path, max_pad_len=174):
        audio, sample_rate = librosa.load(file_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=12) # extract 12 MFCC: the standard is 12-13
        pad_width = max_pad_len - mfcc.shape[1]
        if pad_width > 0:
            mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc = mfcc[:, :max_pad_len]
        return mfcc

# Function to plot MFCCs
def plot_mfcc(mfcc_features, figsize=(10, 6), cmap="coolwarm"):
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the heatmap
    im = ax.imshow(mfcc_features, aspect='auto', origin='lower', cmap=cmap)

    # Add labels and colorbar
    ax.set_title("MFCC Heatmap")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("MFCC Coefficients")
    fig.colorbar(im, ax=ax, orientation="vertical", label="Amplitude")

    # Show the plot in Streamlit
    st.pyplot(fig)

# Load your pre-trained CNN model
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model("RAVDESS_MFCC_Sentiment_Analysis.keras")  # Replace with your model path

cnn_model = load_cnn_model()

# Prompt User to Record Audio
st.header("Record Your Audio")
audio_data = st.audio_input("Record a voice message")

# Process Recorded Audio
if audio_data is not None:
    # Save audio file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio_file:
        temp_audio_file.write(audio_data.getvalue())
        temp_audio_file_path = temp_audio_file.name

    # Playback recorded audio success
    st.success("Audio successfully recorded!")

    st.header("Transcription and Sentiment Analysis")
    
    # Transcription using Whisper
    with st.spinner(text="Transcribing in progress..."):
        transcription_result = whisper_model.transcribe(temp_audio_file_path)
    transcription = transcription_result.get("text", "")
    st.subheader("Transcription using Whisper:")
    st.write(transcription)

    # Sentiment Analysis using NLTK
    sentiment_scores = sentiment_analyzer.polarity_scores(transcription)
    st.subheader("Sentiment Analysis using NLTK:")
    st.json(sentiment_scores)

    # Extract MFCCs
    mfccs = extract_mfcc(temp_audio_file_path)
    st.subheader("MFCC Feature Extraction using Librosa:")
    st.write(mfccs)
    plot_mfcc(mfccs)

    # CNN Sentiment Analysis
    st.header("CNN Model Sentiment Prediction")
    if mfccs is not None:
        mfccs_input = np.expand_dims(mfccs, axis=0)

        prediction = cnn_model.predict(mfccs_input)
        st.subheader("Prediction Result:")
        st.write(prediction)
        if np.argmax(prediction) == 0:
            st.markdown("The audio file is __negative__!")
        elif np.argmax(prediction) == 1:
            st.markdown("The audio file is __neutral__!")
        elif np.argmax(prediction) == 2:
            st.markdown("The audio file is __positive__!")

else:
    st.warning("Please record an audio sample to proceed.")
