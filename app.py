import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.subheader("Model Performance")

if st.checkbox("Show Confusion Matrix"):
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    st.pyplot(fig)

# Load model & vectorizer
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.set_page_config(page_title="Fake News Detector")

st.title("📰 Fake News Detection System")

st.write("Enter news text below to check whether it is REAL or FAKE")

# Text input
user_input = st.text_area("Enter News Content")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)

        if prediction[0] == 1:
            st.success("✅ This news is REAL")
        else:
            st.error("❌ This news is FAKE")


  
