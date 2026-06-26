import streamlit as st
import pickle as pkl
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import matplotlib.pyplot as plt
import seaborn as sns
@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('punkt_tab')

download_nltk_data()
ps=PorterStemmer()

tfidf=pkl.load(open('vectorizer.pkl','rb'))
model=pkl.load(open('model.pkl','rb'))
metrics=pkl.load(open('metrics.pkl','rb'))

tab1,tab2,tab3=st.tabs(['Spam Classifier','Model Metrics','Info'])
with tab1:
    st.title("📩 SMS Spam Classifier")

    st.write("Detect whether a message is Spam or Not Spam")
    input_sms=st.text_input("Enter the message")



    def transform_text(text):
        text=text.lower()
        text=nltk.word_tokenize(text)
        y=[]
        for i in text:
            if i.isalnum():
                y.append(i)
        text=y[:]
        y.clear()
        for i in text:
            if i not in stopwords.words('english') and (i not in string.punctuation):
                y.append(i)
        text=y[:]
        y.clear()
        for i in text:
            y.append(ps.stem(i))
        return ' '.join(y)

    if st.button("Predict"):
        
        if input_sms=="":
            st.warning("Please enter a message!")
            #1 Preprocess
        else:
            transformed_sms=transform_text(input_sms)

            #2 Vectorize
            vector_input=tfidf.transform([transformed_sms])

            #3 Predict
            result=model.predict(vector_input)[0]

        #4 Display
            if result==1:
                st.header("Spam")
            else:
                st.header("Not Spam")
with tab2:
    st.title("Model Performance Metrics")
  
    try:
        metrics = pkl.load(open('metrics.pkl', 'rb'))
         
        metrics_available = True
    except:
        metrics_available = False
    try:
        cm=metrics['confusion_matrix']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        with col2:
            st.metric("Precision", f"{metrics['precision']:.2%}")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.2%}")
        with col4:
            st.metric("F1-Score", f"{metrics['f1']:.2%}")
        st.subheader("Confusion-Matrix")
        fig,ax=plt.subplots()
        sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
        st.pyplot(fig)
    except:
        st.warning("Metrics not found")
with tab3:
    st.title("About this Project")
   
    st.write("""
    ### Model Details
    - **Algorithm**: Multinomial Naive Bayes
    - **Vectorizer**: TF-IDF
    - **Dataset**: SMS Spam Collection
    
    ### How It Works
    1. Text Preprocessing (lowercase, tokenization, stopwords removal)
    2. TF-IDF Vectorization
    3. Naive Bayes Classification
    
    ### Links
    - 📂 [GitHub](https://github.com/codebyprathamesh/SMS-Spam-Classifier)
   
    """)
        




            
            
                
            
            



