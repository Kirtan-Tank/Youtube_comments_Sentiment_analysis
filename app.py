import streamlit as st
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load the saved model
model = load_model('sentiment_model.h5')

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('label_encoder.npy')

# Load the tokenizer
tokenizer = Tokenizer()
tokenizer.word_index = np.load('tokenizer.npy', allow_pickle=True).item()

st.title('Sentiment Analysis App')

# Create a text input for user to enter custom text
user_input = st.text_input("Enter your text:")

if user_input:
    # Preprocess the user input
    custom_text_sequences = tokenizer.texts_to_sequences([user_input])
    custom_text_padded = pad_sequences(custom_text_sequences, maxlen=100, padding='post', truncating='post')

    # Make prediction
    prediction = model.predict(custom_text_padded)
    predicted_sentiment = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    st.write(f'Predicted Sentiment: {predicted_sentiment}')

    if predicted_sentiment == 'negative':
        st.write("Negative sentiment")
    elif predicted_sentiment == 'neutral':
        st.write("Neutral sentiment")
    else:
        st.write("Positive sentiment")