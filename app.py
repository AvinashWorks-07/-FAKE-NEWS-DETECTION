import streamlit as st
import pickle

# Page config
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# Load model safely
@st.cache_resource
def load_model():
    model = pickle.load(open("fake_news_model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_model()

# Sidebar
st.sidebar.title("🧠 Fake News Detector")
st.sidebar.markdown("""
This app uses **Machine Learning**  
to detect **Fake vs Real News**.

📌 Model: Logistic Regression  
📌 Vectorizer: TF-IDF  
""")

st.sidebar.info("Created by Avinash 🚀")

# Main UI
st.markdown("<h1 style='text-align: center;'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Enter a news article below and check its authenticity.</p>", unsafe_allow_html=True)

text = st.text_area("📝 Paste News Content Here", height=200)

if st.button("🔍 Check News"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text!")
    else:
        prediction = model.predict(vectorizer.transform([text]))
        probability = model.predict_proba(vectorizer.transform([text]))[0]

        if prediction[0] == 1:
            st.success("✅ This news appears to be REAL")
            st.progress(probability[1])
            st.write(f"Confidence: **{probability[1]*100:.2f}%**")
        else:
            st.error("❌ This news appears to be FAKE")
            st.progress(probability[0])
            st.write(f"Confidence: **{probability[0]*100:.2f}%**")

# Footer
st.markdown("---")
st.markdown(
    "<center>💡 Built using Machine Learning & Streamlit</center>",
    unsafe_allow_html=True
)
