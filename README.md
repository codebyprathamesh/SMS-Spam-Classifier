# 📩 SMS Spam Classifier

A Machine Learning and NLP-based web application that classifies SMS messages as **Spam** or **Not Spam** using TF-IDF Vectorization and Multinomial Naive Bayes.

## 🌐 Live Demo
[Click here to try the app](https://sms-spam-classifier-10.streamlit.app)

## 📊 Dataset
- Source: SMS Spam Collection Dataset (Kaggle)
- Total messages: 5,572
- After cleaning: 5,169
- Ham (Not Spam): 87.4%
- Spam: 12.6%

## 🔍 EDA Insights
- Data is imbalanced — 87% ham vs 13% spam
- Spam messages are significantly longer than ham
- Avg spam message length: 137 characters vs 70 for ham
- Spam messages contain more words (avg 27) vs ham (avg 17)
- Most common spam words: free, call, claim, prize, win

## ⚙️ Text Preprocessing Pipeline
1. Lowercase conversion
2. Tokenization (NLTK)
3. Removing special characters
4. Removing stopwords and punctuation
5. Stemming (Porter Stemmer)

## 🤖 Model
- **Algorithm:** Multinomial Naive Bayes
- **Vectorizer:** TF-IDF (max_features=3000)
- **Accuracy:** 97.09%
- **Precision:** 100%
- **Confusion Matrix:**
  - True Negatives: 896
  - False Positives: 0
  - False Negatives: 30
  - True Positives: 108

## 🛠️ Tech Stack
- Python
- Pandas, NumPy
- NLTK
- Scikit-learn
- Matplotlib, Seaborn
- WordCloud
- Streamlit

## 📁 Files
- `Spam_msg_detection.ipynb` — Main notebook
- `sms_class.py` — Streamlit web app
- `model.pkl` — Trained MNB model
- `vectorizer.pkl` — TF-IDF vectorizer
- `spam.csv` — Dataset
- `requirements.txt` — Dependencies

## 🚀 How to Run Locally
1. Clone the repo
2. Install requirements: