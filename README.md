# SPAM_DETECTION_


**Description:**


The Spam Detection System is a machine learning project aimed at identifying and filtering out spam emails. 
Leveraging natural language processing (NLP) techniques and supervised learning algorithms, the system 
analyzes email content to differentiate between legitimate messages and spam. By training on labeled datasets containing examples of both spam and non-spam emails, the model learns to recognize patterns indicative 
of spam behavior.


**Data Preprocessing:**


Clean and preprocess email data to extract relevant features and convert text into a format suitable for machine learning algorithms.
Feature Extraction: Utilize NLP techniques to extract features such as word frequency, presence of specific keywords, and syntactic structures from email content.
Model Training: Train a supervised learning model (e.g., Naive Bayes, Support Vector Machine, or Neural Network) using labeled datasets to classify emails as spam or non-spam.
Model Evaluation: Evaluate the performance of the trained model using metrics such as accuracy, precision, recall, and F1-score to assess its effectiveness in detecting spam.
Deployment: Integrate the trained model into an application or service, allowing users to submit emails for spam detection in real-time.
Continuous Improvement: Implement mechanisms for feedback collection to continuously improve the model's accuracy and effectiveness over time.

**Technologies Used:**

Python

Scikit-learn

Natural Language Toolkit (NLTK)

###################################################################################################################################################################################################################


#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import pandas as pd


# In[5]:


df = pd.read_csv("spam.csv")


# In[12]:





# In[ ]:


# 1. data cleaning
# 2. EDA
# 3. text preprocessing
# 4. model building
# 5. Evaluation
# 6. improvement
# 7. website


# # Data cleaning

# In[6]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace = True)


# In[7]:


df.rename(columns={'v1':'check','v2':'message'},inplace = True)


# In[8]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[9]:


df["check"]=encoder.fit_transform(df["check"])


# In[10]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[12]:


df = df.drop_duplicates(keep='first')
df


# # EDA

# In[14]:


df['check'].value_counts()


# In[15]:


import matplotlib.pyplot as plt
plt.pie(df['check'].value_counts(), labels=['ham','spam'],autopct="%0.4f")
plt.show()


# In[16]:


import nltk as nk
nk.download('punkt')


# In[20]:


df['no._char']=df['message'].apply(len)


# In[21]:


df


# In[22]:


df['no._words']= df['message'].apply(lambda x: len(nk.word_tokenize(x)))


# In[23]:


df['no._sentence']= df['message'].apply(lambda x: len(nk.sent_tokenize(x)))


# In[24]:


df


# In[26]:


df.describe()


# In[27]:


#ham
df[df['check']==0][["no._char","no._words","no._sentence"]].describe()


# In[28]:


#spam
df[df['check']==1][["no._char","no._words","no._sentence"]].describe()


# In[29]:


plt.figure(figsize=(24,12))
plt.hist(df[df['check']==0]['no._char'])
plt.hist(df[df['check']==1]['no._char'],color="red")
plt.show()


# In[30]:


plt.figure(figsize=(24,12))
import seaborn as sn
sn.histplot(df[df['check']==0]['no._char'])
sn.histplot(df[df['check']==1]['no._char'],color='red')


# In[31]:


plt.figure(figsize=(24,12))
sn.histplot(df[df['check']==0]['no._words'])
sn.histplot(df[df['check']==1]['no._words'],color='red')


# In[32]:


sn.pairplot(df,hue="check")


# In[118]:


sn.heatmap(df.corr(),annot=True)


# # data pre-processing
# ⚫lower case
# ⚫tokenization
# ⚫removing special char
# ⚫removing stop wrds and punctuation
# ⚫stemming
# 

# In[33]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[34]:


import string as str
str.punctuation


# In[35]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[36]:


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


# In[37]:


df['trans_txt']=df['message'].apply(trans_text)


# In[38]:


df


# In[39]:


from wordcloud import WordCloud
wc = WordCloud(width=5000,height=5000,min_font_size=10,background_color='white')


# In[40]:


spam_wc = wc.generate(df[df['check']==1]['trans_txt'].str.cat(sep=" "))


# In[41]:


plt.imshow(spam_wc)


# In[42]:


ham_wc = wc.generate(df[df['check']==0]['trans_txt'].str.cat(sep=" "))


# In[43]:


plt.imshow(ham_wc)


# In[44]:


spam_corp = []
for t in df[df['check']==1]['trans_txt'].tolist():
    for w in t.split():
        spam_corp.append(w)


# In[45]:


len(spam_corp)


# In[46]:


from collections import Counter
sn.barplot(x=pd.DataFrame(Counter(spam_corp).most_common(30))[0],y=pd.DataFrame(Counter(spam_corp).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[47]:


ham_corp = []
for t in df[df['check']==0]['trans_txt'].tolist():
    for w in t.split():
        ham_corp.append(w)


# In[48]:


len(ham_corp)


# In[49]:


sn.barplot(x=pd.DataFrame(Counter(ham_corp).most_common(30))[0],y=pd.DataFrame(Counter(ham_corp).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# # model build

# In[62]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tf = TfidfVectorizer()


# In[63]:


X = tf.fit_transform(df['trans_txt']).toarray()


# In[64]:


X.shape


# In[65]:


y = df['check'].values


# In[66]:


y


# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)


# In[69]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[70]:


gb = GaussianNB()
mb = MultinomialNB()
bb = BernoulliNB()


# In[71]:


gb.fit(X_train,y_train)
y_pred1 = gb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[72]:


mb.fit(X_train,y_train)
y_pred2 = mb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[73]:


bb.fit(X_train,y_train)
y_pred3 = bb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[ ]:


#tf with mb


# In[74]:


import pickle
pickle.dump(tf,open('vectorizer.pkl','wb'))
pickle.dump(mb,open('model.pkl','wb'))


# In[ ]:





