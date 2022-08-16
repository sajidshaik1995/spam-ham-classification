#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


df = pd.read_csv('spam.csv',encoding="latin-1")


# In[3]:


df.head()


# In[4]:


df.sample(5)


# In[5]:


df.shape


# In[6]:


df.info()


# In[7]:


df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)


# In[8]:


df.head()


# In[9]:


df.sample(5)


# In[10]:


#renames the columns
df.rename(columns={'v1':'target','v2':'text'},inplace=True)


# In[11]:


df.head()


# In[12]:


from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()


# In[13]:


df['target'] = encoder.fit_transform(df['target'])


# In[14]:


df.head()


# In[15]:


df.isnull().sum()


# In[16]:


df.duplicated().sum()


# In[17]:


# remove duplicates
df=df.drop_duplicates(keep='first')


# In[18]:


df.duplicated().sum()


# In[19]:


df.shape


# # EDA

# In[20]:


df['target'].value_counts()


# In[21]:


plt.pie(df['target'].value_counts(), labels=['ham','spam'],autopct="%0.2f")
plt.show()


# In[22]:


## Data is imbalanced


# In[23]:


get_ipython().system(' pip install nltk')


# In[24]:


import nltk


# In[25]:


nltk.download('punkt')


# In[26]:


df['num_characters'] =df['text'].apply(len)


# In[27]:


df.head()


# In[28]:


# number of words
df['num_words']=df['text'].apply(lambda x:len(nltk.word_tokenize(x)))


# In[29]:


df.head()


# In[30]:


df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[31]:


df['num_sentences']=df['text'].apply(lambda x:len(nltk.sent_tokenize(x)))


# In[32]:


df.head()


# In[33]:


df[['num_characters','num_words','num_sentences']].describe()


# In[34]:


## ham messgaes


# In[35]:


df[df['target'] ==0][['num_characters','num_words','num_sentences']].describe()


# In[36]:


## spam messages


# In[37]:


df[df['target'] ==1][['num_characters','num_words','num_sentences']].describe()


# In[38]:


df[df['target'] ==0]['num_characters']


# In[39]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['target'] ==0]['num_characters'])
sns.histplot(df[df['target'] ==1]['num_characters'],color='red')


# In[40]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['target'] ==0]['num_words'])
sns.histplot(df[df['target'] ==1]['num_words'],color='red')


# In[41]:


plt.figure(figsize=(12,8))
sns.histplot(df[df['target'] ==0]['num_sentences'])
sns.histplot(df[df['target'] ==1]['num_sentences'],color='red')


# In[42]:


sns.pairplot(df,hue='target')


# In[43]:


sns.heatmap(df.corr(),annot=True)


# # Data preprocessing
# ## .Lower case
# ## .Tokenization
# ## .Removing special characters
# ## .Removing stopwords and punctuation
# ## .Stemming

# In[44]:


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y =[]
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text =y[:]
    y.clear()
    
    for  i in text:
        if i not in stopwords.words('english') and i  not  in string.punctuation:
            y.append(i)
            
    text =y[:]
    y.clear()
    
    for  i in text:
        y.append(ps.stem(i))
            
    return " ".join(y)


# In[45]:


from nltk.corpus import stopwords
stopwords.words('english')


# In[46]:


import string
string.punctuation


# In[49]:


transform_text("But i'll b going 2 sch on mon. My sis need 2 take smth.")


# In[48]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
ps.stem('History')


# In[50]:


df['transformed_text']=df['text'].apply(transform_text)


# In[51]:


df.head()


# In[52]:


get_ipython().system(' pip install wordcloud')


# In[53]:


from wordcloud import WordCloud
wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')


# In[54]:


spam_wc=wc.generate(df[df['target']== 1]['transformed_text'].str.cat(sep=" "))


# In[55]:


plt.figure(figsize=(15,15))
plt.imshow(spam_wc)


# In[56]:


ham_wc=wc.generate(df[df['target']== 0]['transformed_text'].str.cat(sep=" "))


# In[57]:


ham_wc


# In[58]:


plt.figure(figsize=(15,15))
plt.imshow(ham_wc)


# In[59]:


spam_corpus = []
for msg in df[df['target'] == 1]['transformed_text'].tolist():
    for word in msg.split():
        spam_corpus.append(word)


# In[60]:


len(spam_corpus)


# In[61]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[62]:


ham_corpus = []
for msg in df[df['target'] == 0]['transformed_text'].tolist():
    for word in msg.split():
        ham_corpus.append(word)


# In[63]:


len(ham_corpus)


# In[64]:


from collections import Counter
sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
plt.xticks(rotation='vertical')
plt.show()


# In[65]:


df.head()


# In[66]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
cv = CountVectorizer()
tfidf = TfidfVectorizer(max_features=3000)


# In[67]:


X = tfidf.fit_transform(df['transformed_text']).toarray()


# In[68]:


X.shape


# In[69]:


y = df['target'].values


# In[70]:


y


# In[71]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


# In[72]:


from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score


# In[73]:


gnb = GaussianNB()
mnb = MultinomialNB()
bnb = BernoulliNB()


# In[74]:


gnb.fit(X_train,y_train)
y_pred1 =gnb.predict(X_test)
print(accuracy_score(y_test,y_pred1))
print(confusion_matrix(y_test,y_pred1))
print(precision_score(y_test,y_pred1))


# In[75]:


mnb.fit(X_train,y_train)
y_pred2 =mnb.predict(X_test)
print(accuracy_score(y_test,y_pred2))
print(confusion_matrix(y_test,y_pred2))
print(precision_score(y_test,y_pred2))


# In[76]:


bnb.fit(X_train,y_train)
y_pred3 =bnb.predict(X_test)
print(accuracy_score(y_test,y_pred3))
print(confusion_matrix(y_test,y_pred3))
print(precision_score(y_test,y_pred3))


# In[77]:


# tfidf = mnb 


# In[78]:


import pickle
pickle.dump(tfidf,open('vectorizer.pkl','wb'))
pickle.dump(mnb,open('model.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




