#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense,Activation,Embedding,Flatten,LeakyReLU,BatchNormalization
from keras.activations import relu,sigmoid


# In[2]:


data = pd.read_csv("Churn_Modelling.csv")
data.head()


# In[3]:


x = data.iloc[:,3:13]
y = data.iloc[:,13]


# In[4]:


geography = pd.get_dummies(x['Geography'],drop_first=True)
gender = pd.get_dummies(x['Gender'],drop_first=True)
x = pd.concat([x,geography,gender],axis =1)
x = x.drop(['Geography','Gender'],axis=1)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state = 0)


# In[5]:


stdsclr = StandardScaler()
x_train  = stdsclr.fit_transform(x_train)
x_test = stdsclr.transform(x_test)


# In[6]:


def create_model(layers,activation):
    model = Sequential()
    for i,nodes in enumerate(layers):
        if i == 0 :
            model.add(Dense(nodes,input_dim = x_train.shape[1])
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation.relu))
            model.add(Dropout(0.3))
    model.add(Dense(1,kernel_initializer='glort_uniform',activation='sigmoid'))
    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
    return model


# In[ ]:


model = KerasClassifier(build_fn=create_model,verbose=0)
layers = [[20],[40,20],[45,30,15]]
activations  = ['sigmoid','relu']
para_grid = dict(layers=layers,activations=relu,batch_size = [128,256],epochs=[30])
grid =  GridSearchCV(estimator=model,param_grid=para_grid,cv=5)


# In[ ]:


grid_res = grid.fit(x_train,y_train)


# In[ ]:




