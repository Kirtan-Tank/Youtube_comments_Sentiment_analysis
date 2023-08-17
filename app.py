import streamlit as st
from streamlit_lottie import st_lottie
import json
import pandas as pd
import numpy as np
import requests
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

# st.title('Sentiment Analysis App')
st.columns(3)[1].title("𝐂𝐨𝐦𝐦𝐞𝐧𝐭𝐕𝐢𝐛𝐞𝐬")

st.columns(3)[1].markdown("~ ~ _Analyzing Comment Emotions_")

# Animation with lottie (loading gif files throgh url)
def lottieurl_load(url: str):
    r= requests.get(url)
    if r.status_code !=200:
        return None
    return r.json()
    
lottie_img = lottieurl_load("https://lottie.host/beed94bb-bf7d-470e-a14d-b8c4989260ce/P1cvwIXaxC.json")   
with st.columns(3)[1]:
    st_lottie(lottie_img,speed=1,reverse=False,loop=True,quality="medium",height=250,width=250,key=None)


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
        # st.write("☹️")
        lottie_img_neg = lottieurl_load("https://lottie.host/4440e505-65be-4ae1-8430-5eedfe2d8586/GKwBf1NDeW.json")
        st_lottie(lottie_img_neg,speed=1,reverse=False,loop=True,quality="medium",height=250,width=250,key=None)

    elif predicted_sentiment == 1.0:
        st.write("Neutral sentiment")
        # st.write("😐")
        lottie_img_neut = lottieurl_load("https://lottie.host/146ada86-c643-47a1-8e22-9f90d7b8ab17/5kAs5bkcXN.json")
        st_lottie(lottie_img_neut,speed=1,reverse=False,loop=True,quality="medium",height=250,width=250,key=None)
    else:
        st.write("Positive sentiment")
        # st.write("🙂")
        lottie_img_pos = lottieurl_load("https://lottie.host/92d72349-15d0-4421-a3c6-88a9f8f1aed2/dRMzaFPt8E.json")
        st_lottie(lottie_img_pos,speed=1,reverse=False,loop=True,quality="medium",height=250,width=250,key=None)
