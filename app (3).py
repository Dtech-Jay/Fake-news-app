import streamlit as st
import joblib

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

**How to use:**
1. Enter a statement  
2. Click **Predict**  
3. View the result and confidence
""")

st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 ML Mini Project")

# ---------------- MAIN TITLE ----------------
st.title("📰 Fake News Detection System")
st.markdown("Check whether a statement is **True or Fake** using ML")

# ---------------- LOAD MODEL ----------------
try:
    model = joblib.load("logistic_regression_model.joblib")
    tfidf = joblib.load("tfidf_vectorizer.joblib")
except Exception as e:
    st.error("❌ Model files not found or failed to load.")
    st.stop()

# ---------------- USER INPUT ----------------
user_statement = st.text_area(
    "✍️ Enter the statement:",
    height=150,
    placeholder="Example: The government announced a new education policy today..."
)

# ---------------- PREDICT BUTTON ----------------
if st.button("🔍 Predict"):
    if user_statement.strip() == "":
        st.warning("⚠️ Please enter a statement.")
    else:
        with st.spinner("Analyzing statement..."):
            # 🚨 DO NOT preprocess here (VERY IMPORTANT)
            vector = tfidf.transform([user_statement])
            prediction = model.predict(vector)[0]
            confidence = model.predict_proba(vector).max()

        # ---------------- LABEL MAPPING ----------------
        label_map = {
            0: "❌ False",
            1: "✅ True",
            2: "⚖️ Half True",
            3: "✔️ Mostly True",
            4: "🔥 Pants on Fire"
        }

        # ---------------- RESULT ----------------
        st.markdown("---")
        st.subheader("📊 Prediction Result")
        st.success(f"**Label:** {label_map.get(prediction, prediction)}")
        st.info(f"**Confidence:** {round(confidence * 100, 2)}%")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("📌 This tool is for educational purposes only.")
