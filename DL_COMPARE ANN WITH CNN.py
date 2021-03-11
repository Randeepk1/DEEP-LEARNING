#!/usr/bin/env python
# coding: utf-8

# In[41]:


import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score,classification_report


# In[2]:


(x_train,y_train),(x_test,y_test) =datasets.cifar10.load_data()
x_train.shape


# In[3]:


x_test.shape


# In[4]:


x_train[0]


# In[5]:


plt.imshow(x_train[0])


# In[6]:


y_train.shape


# In[7]:


y_train[:5]


# In[8]:


y_train = y_train.reshape(-1,)
y_train[:5]


# In[9]:


classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


# In[10]:


classes[9]


# In[11]:


def plot_sample(X,y, index):
    plt.figure(figsize=(15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])


# In[12]:


plot_sample(x_train,y_train,0)


# In[13]:


plot_sample(x_train,y_train,3)


# In[14]:


x_train = x_train/255
x_test = x_test/255


# In[15]:


ANN = models.Sequential([
        layers.Flatten(input_shape= (32,32,3)),
        layers.Dense(3000,activation = 'relu'),
        layers.Dense(10,activation='sigmoid')
])

ANN.compile(optimizer = 'SGD',
           loss = 'sparse_categorical_crossentropy',
           metrics = ['accuracy'])
ANN.fit(x_train,y_train,epochs=5)


# In[16]:


ANN.evaluate(x_test,y_test)


# In[18]:


CNN = models.Sequential([
    layers.Conv2D(filters=32,kernel_size=(3, 3), activation ='relu',input_shape=(32,32,3)),
    layers.MaxPool2D((2,2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3),activation='relu'),
    layers.MaxPooling2D((2,2)),
    
    #Dense
    layers.Flatten(),
    layers.Dense(64,activation='relu'),
    layers.Dense(10,activation = 'softmax')
])


# In[23]:


CNN.compile(optimizer = 'adam',
             loss = 'sparse_categorical_crossentropy',
             metrics = ['accuracy'])


# In[24]:


CNN.fit(x_train,y_train,epochs=10)


# In[25]:


CNN.evaluate(x_test,y_test)


# In[26]:


y_test = y_test.reshape(-1,)


# In[27]:


y_test[:5]


# In[28]:


plot_sample(x_test,y_test,1)


# In[29]:


y_pred = CNN.predict(x_test)


# In[30]:


y_pred[:5]


# In[31]:


np.argmax([5,12,167,2])


# In[33]:


y_classes = [np.argmax(element) for element in y_pred]


# In[34]:


y_classes[:5]


# In[35]:


y_test[:5]


# In[39]:


plot_sample(x_test,y_test,3)


# In[37]:


classes


# In[40]:


classes[y_classes[3]]


# In[42]:


print("classification Report :\n",classification_report(y_test,y_classes))

