import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

import streamlit as st
import pickle as pkl
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()

tfidf=pkl.load(open('vectorizer.pkl','rb'))
model=pkl.load(open('model.pkl','rb'))
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




        
        
            
        
        



