import streamlit as st
import pickle

# MUST be first Streamlit command
st.set_page_config(page_title="Fake News Detection")

# Load model and vectorizer safely
@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

st.title("📰 Fake News Detection System")
st.write("Enter a news article to check whether it is REAL or FAKE.")

text = st.text_area("Enter news text")

if st.button("Predict"):
    if text.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = model.predict(vectorizer.transform([text]))
        if prediction[0] == 1:
            st.success("✅ REAL NEWS")
        else:
            st.error("❌ FAKE NEWS")
