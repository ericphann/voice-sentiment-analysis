# 🗣️ Voice Sentiment Analysis
#### Final Project for DSBA 6156: Applied Machine Learning  
##### Team: Eric Phann & Jessica Ricks  
Experiment analyzing various approaches to voice sentiment analysis.  

## 📝 Approach #1: Speech-to-Text / Transcription  
Transcribing the audio sample to text, embedding the text, and training a model to classify sentiment.  

__Pre-trained Wav2Vec Model__: [Wav2Vec2-Large-960h](https://huggingface.co/facebook/wav2vec2-large-960h)  
Wav2Vec is trained using connectionist temporal classification and its outputs must be decoding using a CTC tokenizer.  
Here is a great explanation of [sequence modeling using CTC](https://distill.pub/2017/ctc/).

## 🔉 Approach #2: Mel-Frequency Cepstral Coefficients  
Extract features from the audio sample using Mel-Frequency Cepstral coefficients and training a model to classify sentiment.

## ✨ Approach #3: Whisper by OpenAI  
Using OpenAI's Whisper to transcribe text and prompt ChatGPT to classify sentiment.
