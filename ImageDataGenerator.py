#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras_preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img


# In[2]:


data_gen = ImageDataGenerator(rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.02, shear_range=0.2,
    zoom_range=0.2,horizontal_flip=True,
    vertical_flip=True,fill_mode='nearest',)


# In[3]:


img  = load_img('Randeepk.jpg')


# In[4]:


x= img_to_array(img)  # numpy array with shape(3,720,720)
x.shape
X = x.reshape((1,)+x.shape)  # numpy array with shape(1,3,720,720) # 1 means 4 dimentional
X.shape


# In[ ]:


i=0
for batch in data_gen.flow(X,batch_size=1,save_to_dir='Preview',save_prefix='Randeep',save_format='jpg'):
    i+=1
    if i >20:
        break

