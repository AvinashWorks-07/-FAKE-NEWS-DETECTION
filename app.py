import streamlit as st
import pickle

# MUST be first Streamlit command
st.set_page_config(page_title="Fake News Detection")

# Load model and vectorizer safely
@st.cache_resource
def load_model():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# UI
st.title("📰 Fake News Detection System")
st.write("Enter a news article below to check whether it is REAL or FAKE.")

user_input = st.text_area("Enter News Text")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        prediction = model.predict(vectorizer.transform([user_input]))
        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")
