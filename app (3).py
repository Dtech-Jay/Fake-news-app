import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download necessary NLTK data (only if not already downloaded)
# These lines are crucial for Streamlit deployment as NLTK data might not be present
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
# Initialize lemmatizer and load stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = text.split()   # ✅ SAFE replacement
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return ' '.join(processed_tokens)

# Load the trained model and TF-IDF vectorizer
# Ensure these files are in the same directory as app.py or provide full paths
try:
    model = joblib.load('logistic_regression_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
    st.success("Model and vectorizer loaded successfully!")
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found. Make sure 'logistic_regression_model.joblib' and 'tfidf_vectorizer.joblib' are in the correct directory.")
    st.stop() # Stop the app if files are not found

# Streamlit application layout
st.title("Statement Classification App")
st.write("Enter a statement below to classify its label.")

# Input field for user statement
user_statement = st.text_input("Your statement:", "")

if user_statement:
    st.write("Processing your statement...")
    # Preprocess the user's statement
    preprocessed_statement = preprocess_text(user_statement)
    st.write(f"Preprocessed statement: {preprocessed_statement}")

    # Transform the preprocessed statement using the loaded TF-IDF vectorizer
    statement_vectorized = tfidf_vectorizer.transform([preprocessed_statement])

    # Make a prediction using the loaded model
    prediction = model.predict(statement_vectorized)

    # Display the prediction
    st.subheader("Prediction:")
    st.write(f"The predicted label for your statement is: {prediction[0]}")


label_map = {
    0: "False",
    1: "True",
    2: "Half True",
    3: "Mostly True",
    4: "Pants on Fire"
}

st.write(f"Prediction: {label_map.get(prediction[0], prediction[0])}")
