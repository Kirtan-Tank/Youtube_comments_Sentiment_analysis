import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
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

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
    

lottie_coding = load_lottiefile("y1.json")  # replace link to local lottie file
lottie_hello = load_lottieurl("https://lottie.host/ba5f1ab0-4983-4652-8d20-7ccbf72c67da/XIs15NIuG8.json")

st_lottie(
    lottie_hello,
    speed=1,
    reverse=False,
    loop=True,
    quality="low", # medium ; high
    renderer="svg", # canvas
    height=None,
    width=None,
    key=None,
)

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

    if predicted_sentiment == 0.0:
        st.write("Negative sentiment ")
        st.write("☹️")
    elif predicted_sentiment == 1.0:
        st.write("Neutral sentiment")
        st.write("😐")
    else:
        st.write("Positive sentiment")
        st.write("🙂")
