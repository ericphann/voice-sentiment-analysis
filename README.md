# 🗣️ Voice Sentiment Analysis
#### Final Project for DSBA 6156: Applied Machine Learning  
##### Team: Eric Phann & Jessica Ricks  
Experiment analyzing various approaches to voice sentiment analysis. We will be using the [RAVDESS dataset](https://zenodo.org/records/1188976) to test our approaches, which is also available on [Kaggle](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio). The goal is to wrap these three approaches in a Streamlit app where users can provide live audio inputs.  

## ✨ Getting Started  
See our [Streamlit App](https://dsba6156-voice-sentiment-analysis.streamlit.app/) and [Final Presentation](https://docs.google.com/presentation/d/18L64Poe5cV0n1BEYlrE9AyrDcyeh2xB7CktCSxRDHHg/edit?usp=sharing) to interact with our project.
If you'd like to learn a bit more about each of the approaches, then we recommend walking through each of the provided notebooks.  
If you'd like to run locally, feel free to clone the repo and `pip install -r requirements.txt` for dependencies. Then `streamlit run demo_app.py`  
_Note: there is a [small bug](https://github.com/streamlit/streamlit/issues/9799) with reproducing the input audio on Firefox. If using Firefox, please install streamlit using the .whl provided._  

## 📝 Approach #1: Transcription Sentiment Analysis
Transcribing audio samples to raw text and running it through sentiment analysis model. Additionally, embedding the raw text and then running sentiment analysis on the embeddings.

We use the pre-trained Wav2Vec Model [Wav2Vec2-Large-960h](https://huggingface.co/facebook/wav2vec2-large-960h) for transcription.  
_Wav2Vec is trained using connectionist temporal classification and its outputs must be decoded using a CTC tokenizer._  
_Here is a great explanation of [sequence modeling using CTC](https://distill.pub/2017/ctc/)._  

We use NLTK's [Sentiment Intensity Analyzer](https://www.nltk.org/api/nltk.sentiment.vader.html) to classify the raw-text transcriptions as positive, neutral, or negative.  
We use the [DistilBERT base uncased finetuned SST-2](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english) model with the HF sentiment analysis pipeline to classify _embedded_ transcriptions as positive or negative.  

We then reclassify emotions from the RAVDESS dataset to develop our ground truths and align to our desired classes of positive, neutral, and negative. Then we evaluate the results of both models.

## 🔉 Approach #2: Mel-Frequency Cepstral Coefficients  
Extract features from audio samples using Mel-Frequency Cepstral coefficients and train a neural network to classify sentiment as either positive, neutral, or negative.  

After labelling the emotions from the RAVDESS dataset similar to approach #1 we extract 12 MFCCs from each audio file. We then train a convolution neural net (CNN) using the MFCC feature to predict whether the audio file is positive, neutral, or negative.   

The model is available on Hugging Face: [RAVDESS MFCC Sentiment Analysis](ericphann/RAVDESS_MFCC_Sentiment_Analysis).

## ✨ Approach #3: Whisper by OpenAI  
Transcribe the audio files into text using Whisper and running it through text-based sentiment analysis models from approach #1.

## 📊 Evaluation  
__Approach #1__: ~50% accuracy in positive/negative binary classification  
__Approach #2__: ~78% accuracy in positive/neutral/negative multiclassification  

## ✔ Conclusion  
Our experiment shows that, when using the RAVDESS dataset, simply running sentiment analysis on audio transcriptions is not sufficient. As seen in Approach #1, it is no better than guessing the sentiment of the transcription. However, using the audio file and extracting Mel-Frequency Cepstral Coefficients in Approach #2 yielded higher accuracy in classification across 3 classes. This supports the idea that audio files contain dimensions and signal that cannot be captured in simple raw text.
