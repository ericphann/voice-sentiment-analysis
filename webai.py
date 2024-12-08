import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# import liabraries
import streamlit as st
import whisper 
from pydub import AudioSegment
from textblob import TextBlob  # For sentiment analysis

st.title("Welcome! Choose between a audio file or create your own audio file!")
#audio_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])
uploaded_file = st.file_uploader("Upload WAV file", type="wav")


model= whisper.load_model("base")
st.text("Model Loaded")


if st.sidebar.button("Transcribe Audio"):
    if uploaded_file is not None:
        st.sidebar.success("Transcribe Audio")
        transcribe = model.transcribe(uploaded_file.name)
        st.sidebar.success("Tanscription Complete")
        st.markdown(transcribe["text"])
    else:
        st.sidebar.warning("Please upload an audio file")

st.sidebar.header("Play orginal audio file")
st.sidebar.audio(uploaded_file)


# Sentiment Analysis Function
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity  # Ranges from -1 (negative) to 1 (positive)
    if polarity > 0.1:
        return "Positive", polarity
    elif polarity < -0.1:
        return "Negative", polarity
    else:
        return "Neutral", polarity

# Streamlit App
st.title("Audio Transcription and Sentiment Analysis")
st.write("Upload a WAV file to get the transcription and sentiment analysis.")

# File Uploader

if uploaded_file is not None:
    with st.spinner("Processing the audio file..."):
        # Save the uploaded file locally
        wav_file_path = f"uploaded_{uploaded_file.name}"
        with open(wav_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Load Whisper model
        model = whisper.load_model("base")

        # Transcribe audio
        st.write("Transcribing the audio...")
        result = model.transcribe(wav_file_path)
        transcript = result["text"]

        # Analyze sentiment
        sentiment, score = analyze_sentiment(transcript)

        # Display Results
        st.success("Processing Complete!")
        st.subheader("Transcription")
        st.write(transcript)

        st.subheader("Sentiment Analysis")
        st.write(f"**Sentiment:** {sentiment} (Score: {score:.2f})")

        # Optional: Clean up temporary files
        os.remove(wav_file_path)
