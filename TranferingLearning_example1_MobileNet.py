#!/usr/bin/env python
# coding: utf-8

# In[74]:


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from IPython.display import Image    # for loading  image  method 1
from tensorflow.keras.preprocessing import image# for loading image method2
from tensorflow.keras.applications import imagenet_utils
from PIL import Image      # for loading image method3


# In[75]:


filename = 'policevan.jpg'
# img = Image(filename = 'sachin3.jpg',width=224,height=224)
# img


# In[76]:


img1 = image.load_img(filename,target_size=(224,224))
plt.imshow(img1)


# In[77]:


mobile = tf.keras.applications.mobilenet_v2.MobileNetV2()# DL Model weights (pre-trained) loading DL model


# In[78]:


resized_img = image.array_to_img(img1)            # preprocessing image
final_img = np.expand_dims(resized_img,axis=0) # need 4th dimension
final_img = tf.keras.applications.mobilenet_v2.preprocess_input(final_img)
final_img.shape


# In[79]:


pred = mobile.predict(final_img)

print(pred)
# In[80]:


res = imagenet_utils.decode_predictions(pred)


# In[81]:


res


# In[ ]:





# In[ ]:





# In[ ]:




