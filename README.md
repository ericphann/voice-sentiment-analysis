# 🗣️ Voice Sentiment Analysis
#### Final Project for DSBA 6156: Applied Machine Learning  
##### Team: Eric Phann & Jessica Ricks  
Experiment analyzing various approaches to voice sentiment analysis. We will be using the [RAVDESS dataset](https://zenodo.org/records/1188976) to test our approaches, which is also available on [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The goal is to wrap these three approaches in a Streamlit app where users can provide live audio inputs.  

## Getting Started  

## 📝 Approach #1: Speech-to-Text / Transcription  
Transcribing the audio sample to text and running it through a sentiment analysis model.  

__Pre-trained Wav2Vec Model__: [Wav2Vec2-Large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)  
Wav2Vec is trained using connectionist temporal classification and its outputs must be decoding using a CTC tokenizer.  
Here is a great explanation of [sequence modeling using CTC](https://distill.pub/2017/ctc/).

## 🔉 Approach #2: Mel-Frequency Cepstral Coefficients  
Extract features from the audio sample using Mel-Frequency Cepstral coefficients and training a model to classify sentiment.

## ✨ Approach #3: Whisper by OpenAI  
Using OpenAI's Whisper to transcribe text and prompt ChatGPT to classify sentiment.

## Evaluation  

## Conclusion
