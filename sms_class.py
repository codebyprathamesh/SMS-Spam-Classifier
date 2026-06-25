import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import streamlit as st
import pickle as pkl
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns

ps = PorterStemmer()

# Load models
tfidf = pkl.load(open('vectorizer.pkl', 'rb'))
model = pkl.load(open('model.pkl', 'rb'))

# Try to load metrics (optional)
try:
    cm = pkl.load(open('confusion_matrix.pkl', 'rb'))
    metrics = pkl.load(open('metrics.pkl', 'rb'))
    metrics_available = True
except:
    metrics_available = False

st.set_page_config(page_title="SMS Spam Classifier", page_icon="📩")
st.title("📩 SMS Spam Classifier")
st.write("Detect whether a message is Spam or Not Spam using Machine Learning")

# Create tabs
tab1, tab2, tab3 = st.tabs(["Spam Classifier", "Model Metrics", "About"])

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and (i not in string.punctuation):
            y.append(i)
    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))
    return ' '.join(y)

# TAB 1: SPAM CLASSIFIER
with tab1:
    st.info(
        "⚠️ Note: This tool uses Machine Learning, which isn't always 100% accurate. "
        "Results may occasionally be wrong — use your own judgment too!"
    )
    
    input_sms = st.text_input("Enter the message")
    
    if st.button("Predict", key="predict_button"):
        if input_sms == "":
            st.warning("Please enter a message!")
        else:
            # Preprocess
            transformed_sms = transform_text(input_sms)
            
            # Vectorize
            vector_input = tfidf.transform([transformed_sms])
            
            # Predict
            result = model.predict(vector_input)[0]
            
            # Display
            if result == 1:
                st.header("🚨 Spam")
                st.error("This message is classified as **SPAM**")
            else:
                st.header("✅ Not Spam")
                st.success("This message is **NOT SPAM**")

# TAB 2: MODEL METRICS
with tab2:
    st.subheader("Model Performance Metrics")
    
    if metrics_available:
        # Display metric cards
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.2%}")
        
        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                    xticklabels=['Not Spam', 'Spam'], 
                    yticklabels=['Not Spam', 'Spam'],
                    cbar_kws={'label': 'Count'})
        plt.ylabel("Actual", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)
        st.pyplot(fig)
        
        st.subheader("Metrics Explanation")
        st.write("""
        - **Accuracy**: Overall correctness of the model
        - **Precision**: Of all predicted spam, how many were actually spam
        - **Recall**: Of all actual spam, how many did the model catch
        - **F1-Score**: Harmonic mean of precision and recall
        """)
    else:
        st.warning("📊 Metrics files not found. Please add:")
        st.code("""
confusion_matrix.pkl
metrics.pkl (with keys: 'accuracy', 'precision', 'recall', 'f1')
        """)

# TAB 3: ABOUT
with tab3:
    st.subheader("About This Project")
    st.write("""
    ### Model Details
    - **Algorithm**: Multinomial Naive Bayes
    - **Vectorizer**: TF-IDF (Term Frequency-Inverse Document Frequency)
    - **Dataset**: SMS Spam Collection (~5,500 messages)
    
    ### How It Works
    1. **Text Preprocessing**: Lowercase, tokenization, stopword removal, stemming
    2. **Vectorization**: Convert text to numerical TF-IDF vectors
    3. **Classification**: Naive Bayes predicts spam or not spam
    
    ### Tech Stack
    - Python, scikit-learn, NLTK, Streamlit
    
    ### Links
    - 📂 [GitHub Repository](https://github.com/codebyprathamesh/spam-classifier)
    
