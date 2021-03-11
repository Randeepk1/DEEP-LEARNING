#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
from sklearn.metrics import confusion_matrix,accuracy_score


# In[2]:


data = pd.read_csv("Churn_Modelling.csv")


# In[3]:


data


# In[4]:


x = data.iloc[:,3:13]
y = data.iloc[:,13]


# In[5]:


geography = pd.get_dummies(x['Geography'],drop_first=True)


# In[6]:


gender = pd.get_dummies(x['Gender'],drop_first=True)


# In[7]:


x = pd.concat([x,geography,gender],axis =1)


# In[8]:


x = x.drop(['Geography','Gender'],axis=1)


# In[9]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 0)


# In[10]:


x_train.shape


# In[11]:


stdsclr = StandardScaler()


# In[12]:


x_train  = stdsclr.fit_transform(x_train)


# In[13]:


x_train.shape


# In[14]:


x_test = stdsclr.transform(x_test)


# In[15]:


x_test.shape


# In[16]:


clf = Sequential()


# In[17]:


clf.add(Dense(6,kernel_initializer ='he_uniform',activation = "relu",input_dim =11)) # Dense for creating  hidden layer,6 means howmany hidden layres he_uniform for weight initilazation
                                                                                     # when use relu function he_uniform(weight initilasation) have to use,input_dim means imput dimention


# In[18]:


clf.add(Dropout(0.3))


# In[19]:


clf.add(Dense(6,kernel_initializer='he_uniform',activation="relu"))


# In[20]:


clf.add(Dropout(0.4))


# In[21]:


clf.add(Dense(1,kernel_initializer='glorot_uniform',activation="sigmoid"))


# In[22]:


clf.summary()


# In[23]:


clf.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) # binary_crossentropy for binary output


# In[29]:


model_hist = clf.fit(x_train,y_train,validation_split=0.30,batch_size=10,epochs=100)


# In[30]:


model_hist.history.keys()


# In[32]:


plt.plot(model_hist.history['accuracy'])
plt.plot(model_hist.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("epochs")
plt.legend(['train','test'],loc="upper left")
plt.show()


# In[25]:


y_pred = clf.predict(x_test)


# In[26]:


y_pred = (y_pred > 0.5)


# In[27]:


confusion_matrix(y_test,y_pred)


# In[28]:


accuracy_score(y_test,y_pred)*100


# In[33]:


plt.plot(model_hist.history['loss'])
plt.plot(model_hist.history['val_loss'])
plt.title("Model loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.legend(['train','test'],loc="upper left")
plt.show()


# In[ ]:




