#!/usr/bin/env python
# coding: utf-8

# In[33]:


import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import seaborn as sns
from sklearn.metrics import accuracy_score


# In[13]:


data = pd.read_csv('G:\DATASETS\diabetes.csv')


# In[14]:


data.head()


# In[15]:


data['Outcome'].value_counts().plot(kind= 'bar')


# In[16]:


x = data.iloc[:,0:8]
y = data.iloc[:,8]


# In[17]:


x_train,x_test,y_train,y_test  = train_test_split(x,y,test_size = 0.3)


# In[18]:


x_train.shape,x_test.shape,y_train.shape,y_test.shape


# In[26]:


model = Sequential()
model.add(Dense(12,input_dim = 8,activation = 'relu')) # relu is to avoid vanishing problem/exploding gradient problem,,# first hidden layer, input dimention use in first hiden layer only
model.add(Dense(8,activation='relu'))  # dense is fully connected layer
model.add(Dense(1,activation="sigmoid")) # output layer,output is binary inthis case so use sigmoid,sigmoid using for classification problem # its always gives  1 or 0 
# weights and bias are initilization done by keras default method is using 'glorot uniform'


# In[29]:


model.compile(loss="binary_crossentropy",optimizer ='adam',metrics=['accuracy'])  # compile model


# In[30]:


model.fit(x_train,y_train,epochs=150,batch_size = 10)


# In[32]:


_,accuracy = model.evaluate(x_train,y_train)
print('Train Acuracy:%.2f' % (accuracy*100))


# In[36]:


y_pred = model.predict_classes(x_test)


# In[37]:


accuracy_score(y_test,y_pred)

