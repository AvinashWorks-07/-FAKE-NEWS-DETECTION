import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# -------------------- LOAD MODEL --------------------
@st.cache_resource
def load_model():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🧠 Fake News Detector")
st.sidebar.markdown("### Settings")

theme = st.sidebar.radio("Choose Theme", ["Light Mode", "Dark Mode"])

st.sidebar.markdown("---")
st.sidebar.info("👨‍💻 Developed by Avinash")

# Theme effect
if theme == "Dark Mode":
    st.markdown("""
        <style>
        body { background-color: #0E1117; color: white; }
        </style>
    """, unsafe_allow_html=True)

# -------------------- MAIN UI --------------------
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Check whether a news article is REAL or FAKE</p>", unsafe_allow_html=True)

# Input options
option = st.radio("Choose input method:", ["✍️ Paste Text", "📂 Upload .txt File"])

news_text = ""

if option == "✍️ Paste Text":
    news_text = st.text_area("Paste news text here", height=200)

elif option == "📂 Upload .txt File":
    uploaded_file = st.file_uploader("Upload text file", type=["txt"])
    if uploaded_file:
        news_text = uploaded_file.read().decode("utf-8")

# Prediction history
if "history" not in st.session_state:
    st.session_state.history = []

# Predict button
if st.button("🔍 Analyze News"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        vector = vectorizer.transform([news_text])
        prediction = model.predict(vector)
        prob = model.predict_proba(vector)[0]

        if prediction[0] == 1:
            result = "REAL"
            confidence = prob[1]
            st.success(f"✅ REAL NEWS ({confidence*100:.2f}%)")
        else:
            result = "FAKE"
            confidence = prob[0]
            st.error(f"❌ FAKE NEWS ({confidence*100:.2f}%)")

        # Save history
        st.session_state.history.append({
            "Text": news_text[:100] + "...",
            "Prediction": result,
            "Confidence": f"{confidence*100:.2f}%"
        })

        # Confidence chart
        fig, ax = plt.subplots()
        ax.bar(["Fake", "Real"], prob, color=["red", "green"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")
        st.pyplot(fig)

# -------------------- HISTORY --------------------
st.markdown("---")
st.subheader("🕘 Prediction History")

if st.session_state.history:
    st.table(st.session_state.history)
else:
    st.info("No predictions yet.")

# -------------------- FOOTER --------------------
st.markdown(
    "<hr><center>🚀 Built with Streamlit & Machine Learning</center>",
    unsafe_allow_html=True
)
