# Youtube_comments_Sentiment_analysis

This is a simple sentiment analysis application built using Streamlit. It allows users to input a comment, and the app predicts the sentiment of the comment (positive, negative, or neutral). The app also displays a relevant animation based on the predicted sentiment.

## Features

- **Text Input**: Users can input a custom text for analysis.
- **Sentiment Prediction**: The app predicts the sentiment of the text using a pre-trained model.
- **Animations**: Displays a custom animation based on the sentiment of the text (positive, neutral, or negative).
  
## Technologies Used

- **Streamlit**: A Python library for building interactive web applications.
- **Keras**: Used to load the pre-trained sentiment analysis model.
- **Scikit-learn**: Used for label encoding.
- **Lottie**: Used for displaying animations based on the sentiment.
- **Pandas & Numpy**: For data manipulation and numerical computations.
- **Requests**: For fetching Lottie animations.

## Setup Instructions

### Prerequisites

Make sure you have the following dependencies installed:

```bash
pip install streamlit
pip install streamlit-lottie
pip install keras
pip install scikit-learn
pip install numpy
pip install pandas
pip install requests
```

### Files Required

1. **sentiment_model.h5**: The pre-trained sentiment analysis model.
2. **label_encoder.npy**: The saved label encoder used to map the sentiment labels.
3. **tokenizer.npy**: The saved tokenizer to preprocess input text.

### Running the App

1. Place the `sentiment_model.h5`, `label_encoder.npy`, and `tokenizer.npy` files in the same directory as the app script.
2. Run the app using the following command:

```bash
streamlit run app.py
```

3. Open the URL provided by Streamlit in your browser to interact with the application.

## How It Works

1. **User Input**: The user enters a text in the input field.
2. **Text Preprocessing**: The text is tokenized and padded to match the input format expected by the model.
3. **Sentiment Prediction**: The processed text is passed through the pre-trained sentiment model to predict its sentiment.
4. **Display Sentiment**: Based on the prediction, the app displays the predicted sentiment (positive, neutral, or negative) and a corresponding animation.

## Example

- **Input**: "I love this product!"
- **Predicted Sentiment**: Positive
- **Animation**: Happy animation is shown.

## Customization

You can change the Lottie animation URLs to use your own animations or modify the model, tokenizer, and label encoder files to suit your use case.

## License

Please see the 'LICENSE' file for more information.

