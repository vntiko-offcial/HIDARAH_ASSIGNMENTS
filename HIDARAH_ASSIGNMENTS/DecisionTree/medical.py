#!/usr/bin/env python
# coding: utf-8

# In[3]:


#GET BASE KNOWLEDGE 

import pandas as pd
# import pandas to load and process spreadsheet-type data

medical_dataset=pd.read_csv(r'C:\Users\G1\Desktop\medical_data.csv') 

# load a medical dataset.

medical_dataset


# In[4]:


set(medical_dataset['diagnosis'])


# In[5]:



#CREATE MODEL


from sklearn.tree import DecisionTreeClassifier

def diagnose_v4(train_dataset:pd.DataFrame):
    
    # create a Decision Tree Classifier
    model=DecisionTreeClassifier(random_state=1)

    # drop the diagnosis column to get only the symptoms
    train_patient_symptoms=train_dataset.drop(columns=['diagnosis'])

    # get the diagnosis column, to be used as the classification target
    train_diagnoses=train_data['diagnosis']

    # build a decision tree
    model.fit(train_patient_symptoms, train_diagnoses)

    # return the trained model
    return model


# In[11]:



#SPLIT DB  
from sklearn.model_selection import train_test_split

train_data, test_data = train_test_split(medical_dataset, test_size=0.05, random_state=0)
print(train_data.shape)
print(test_data.shape)


# In[12]:



#DRAW DECISION TREE

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
my_tree=diagnose_v4(train_data) # train a model
print(my_tree.classes_) # print the possible target labels (diagnoses)
plt.figure(figsize=(12,6)) # size of the visualization, in inches
# plot the tree
plot_tree(my_tree,
            max_depth=2,
            fontsize=10,
            feature_names=medical_dataset.columns[:-1]
)


# In[13]:


#TEST DATA

from sklearn.metrics import accuracy_score, confusion_matrix

test_patient_symptoms=test_data.drop(columns=['diagnosis'])

test_diagnoses=test_data['diagnosis']

pred=my_tree.predict(test_patient_symptoms)
print(pred)

accuracy_score(test_diagnoses, pred)


# In[15]:


confusion_matrix(test_diagnoses, pred)


# In[ ]:




