📰 Fake News Detection using Machine Learning

This project is a Fake News Detection System built using Machine Learning and deployed using Streamlit.
It classifies news articles as Real or Fake based on textual content.

🚀 Project Overview

Fake news has become a major issue in today’s digital world. This project aims to detect fake news using Natural Language Processing (NLP) and Machine Learning techniques.

The model is trained using TF-IDF Vectorization and Logistic Regression for classification.

🧠 Technologies Used

Python
Pandas
NumPy
Scikit-learn
Streamlit
Matplotlib
Seaborn

📂 Project Structure
FAKE-NEWS-DETECTION/
│
├── app.py                     # Streamlit application
├── fake_news_model.pkl        # Trained ML model
├── vectorizer.pkl             # TF-IDF vectorizer
├── fake_news_model.ipynb      # Model training notebook
├── dataset/
│   ├── Fake.csv
│   └── True.csv
├── requirements.txt
└── README.md

⚙️ How to Run the Project Locally
Step 1: Clone Repository
git clone https://github.com/your-username/FAKE-NEWS-DETECTION.git
cd FAKE-NEWS-DETECTION

Step 2: Install Dependencies
pip install -r requirements.txt

Step 3: Run Streamlit App
streamlit run app.py

🧪 Model Training (Optional)

If you want to retrain the model:

python fake_news_model.py


This will generate:

fake_news_model.pkl

vectorizer.pkl

📊 Dataset Information

Dataset contains two files:

Fake.csv – Fake news articles

True.csv – Real news articles

Source: Kaggle (Fake and Real News Dataset)

📈 Model Performance

Algorithm: Logistic Regression

Vectorization: TF-IDF

Accuracy: ~94%

🌐 Live Demo

Deployed using Streamlit Cloud
(Insert your Streamlit app link here)

🧑‍💻 Author

Name: Avinash Patel
Course: B.Tech (CSE – AI/ML)

📜 License

This project is licensed under the MIT License.

⭐ Support

If you like this project, please ⭐ star the repository!
