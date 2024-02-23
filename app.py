import streamlit as st
import pickle 
import string as str

from nltk.corpus import stopwords
import nltk as nk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def trans_text(text):
    text=text.lower()
    text = nk.word_tokenize(text)
    
    y=[]
    for i in text:
        if i.isalnum():
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in str.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
        
    return " ".join(y)

tf = pickle.load(open("vectorizer.pkl",'rb'))
md = pickle.load(open("model.pkl",'rb'))

st.title("Email/SMS Spam Identifier")
input_msg = st.text_area("Enter the Message")
if st.button("Predict"):
    # pre-process
    trans_msg= trans_text(input_msg)
    # vectorize
    vector_input = tf.transform([trans_msg])
    # predict
    result = md.predict(vector_input)[0]
    # display
    if result == 1:
        st.header("SPAM")
    else:
        st.header("NOT SPAM")    


