import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,SimpleRNN,Dense
from tensorflow.keras.models import load_model

## Load the IMDB dataset word index
word_index=imdb.get_word_index()
#word_index
reverse_word_index = {value: key for key, value in word_index.items()}
# reverse_word_index

## Load the pretained model with ReLu activation
model=load_model('simple_rnn_imdb.h5')

## Step3: Helper Functions
# Function to decode reviews

def decode_reviews(encode_review):
    return ' '.join([reverse_word_index.get(i-3,'?') for i in encode_review])

## Function to preprocess user input

def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word,2)+3 for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

## Creating our prediction function
## Prediction function

def predict_sentiment(review):
    preprocess_input=preprocess_text(review)
    prediction=model.predict(preprocess_input)
    sentiment='Positive' if prediction[0][0]>0.5 else 'negative'
    return sentiment,prediction[0][0]

import streamlit as st
## Design streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative')

## User input
user_input=st.text_area('Movie Review')

if st.button('Classify'):
    preprocess_input=preprocess_text(user_input)

    ## make prediction
    pred=model.predict(preprocess_input)
    sentiment='Positve' if pred[0][0]>0.5 else 'Negative'

    # Display Review
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score:{pred[0][0]}')
else:
    st.write('Please Enter a Movie Review')
