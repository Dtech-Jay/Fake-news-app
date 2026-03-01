import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- SIDEBAR ----------------
st.sidebar.title("ℹ️ About This App")
st.sidebar.write("""
This application uses a **Machine Learning model**
to classify a statement as **True / False / Partially True**.

**Steps:**
1. Enter a statement  
2. Click **Predict**  
3. View result & confidence
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed for ML Mini Project")

# ---------------- MAIN TITLE ----------------
st.title("📰 Fake News Detection System")
st.markdown("### Enter a statement to check its authenticity")

# ---------------- TEXT PREPROCESSING ----------------
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z ]', '', text)
    tokens = text.split()
    processed_tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]
    return ' '.join(processed_tokens)

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("logistic_regression_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
except:
    st.error("❌ Model files not found. Please check deployment.")
    st.stop()

# ---------------- USER INPUT ----------------
user_statement = st.text_area(
    "✍️ Type your statement here:",
    height=150,
    placeholder="Example: The government has announced a new policy today..."
)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict"):
    if user_statement.strip() == "":
        st.warning("⚠️ Please enter a statement before predicting.")
    else:
        with st.spinner("Analyzing statement..."):
            clean_text = preprocess_text(user_statement)
            vector = tfidf.transform([clean_text])

            prediction = model.predict(vector)[0]
            probability = model.predict_proba(vector).max()

        # ---------------- LABEL MAPPING ----------------
        label_map = {
            0: "❌ False",
            1: "✅ True",
            2: "⚖️ Half True",
            3: "✔️ Mostly True",
            4: "🔥 Pants on Fire"
        }

        # ---------------- RESULT DISPLAY ----------------
        st.markdown("---")
        st.subheader("📊 Prediction Result")

        st.success(f"**Label:** {label_map.get(prediction, prediction)}")
        st.info(f"**Confidence:** {round(probability * 100, 2)}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("📌 This tool is for educational purposes only.")
