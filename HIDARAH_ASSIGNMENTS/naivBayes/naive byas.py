#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd


# In[8]:


df = pd.read_csv(r"C:\Users\G1\Desktop\spam.csv")
df.head()


# In[9]:


df.groupby('Category').describe()


# In[10]:


df['spam']= df['Category'].apply(lambda x: 1 if x=='spam' else 0)
df.head()


# In[12]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)


# In[13]:


from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_count = v.fit_transform(X_train.values)
X_train_count.toarray()[:2]


# In[14]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count, y_train)


# In[18]:


emails = [
    "Hey mohan, can we get together to watch footbal game tomorrow?",
    "Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!"
]
emails_count = v.transform(emails)
model.predict(emails_count)


# In[ ]:




