# 🗣️ Voice Sentiment Analysis
#### Final Project for DSBA 6156: Applied Machine Learning  
##### Team: Eric Phann & Jessica Ricks  
Experiment analyzing various approaches to voice sentiment analysis. We will be using the [RAVDESS dataset](https://zenodo.org/records/1188976) to test our approaches, which is also available on [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The goal is to wrap these three approaches in a Streamlit app where users can provide live audio inputs.  

## ✨ Getting Started  
See our [Streamlit App]() and [Final Presentation]() to interact with our project and see a live-demo, respectively.  
If you'd like to learn a bit more about each of the approaches, then we recommend walking through each of the provided notebooks.  

## 📝 Approach #1: Transcription Sentiment Analysis
Transcribing the audio sample to raw text and running it through sentiment analysis model. Additionally, embedding the raw text and then running sentiment analysis on the embeddings.

We use the pre-trained Wav2Vec Model [Wav2Vec2-Large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) for transcription.  
_Wav2Vec is trained using connectionist temporal classification and its outputs must be decoded using a CTC tokenizer._  
_Here is a great explanation of [sequence modeling using CTC](https://distill.pub/2017/ctc/)._  

We use NLTK's [Sentiment Intensity Analyzer](https://www.nltk.org/api/nltk.sentiment.vader.html) to classify the raw-text transcriptions as positive, neutral, or negative.  
We use the [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model with the HF sentiment analysis pipeline to classify _embedded_ transcriptions as positive or negative.  

We then reclassify emotions from the RAVDESS dataset to develop our ground truths and align to our desired classes of positive, neutral, and negative. Then we evaluate the results of both models.

## 🔉 Approach #2: Mel-Frequency Cepstral Coefficients  
Extract features from the audio sample using Mel-Frequency Cepstral coefficients and train a neural network to classify sentiment.

## ✨ Approach #3: Whisper & ChatGPT by OpenAI  
Using OpenAI's Whisper to transcribe text and prompt ChatGPT to classify sentiment.
Transcribing the audio files into text and running the text through ChatGPT to classify whether the text is positive/negative/neutral.

Link to Streamlit App: http://localhost:8501/

## Evaluation  

## Conclusion
