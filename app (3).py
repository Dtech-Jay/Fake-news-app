import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK data (only if not already downloaded)
# These lines are crucial for Streamlit deployment as NLTK data might not be present
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and load stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # 1. Convert to lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = re.sub(r'[^a-z ]', '', text) # Keep only lowercase letters and spaces
    # 3. Tokenize
    tokens = word_tokenize(text)
    # 4. Remove stop words and lemmatize
    processed_tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    # 5. Join back into a string
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
