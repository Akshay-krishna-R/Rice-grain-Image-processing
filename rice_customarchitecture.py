#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image


# In[2]:


from PIL import Image # We use the PIL Library to resize images
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout
data = []
labels = []


# In[3]:


grain_add = os.listdir("../DATA/image aug/broken grains/")
for grain_img in grain_add:
    try:
        image=cv2.imread("../DATA/image aug/broken grains/"+ grain_img)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((224, 224))
        data.append(np.array(size_image))
        labels.append(0)
    except AttributeError:
        print("")


# In[4]:


grain_add


# In[5]:


len(data)


# In[6]:


grain_add = os.listdir("../DATA/image aug/full_grains/")
for grain_img in grain_add:
    try:
        image=cv2.imread("../DATA/image aug/full_grains/"+ grain_img)
        image_from_array = Image.fromarray(image, 'RGB')
        size_image = image_from_array.resize((224, 224))
        data.append(np.array(size_image))
        labels.append(1)
    except AttributeError:
        print("")


# In[7]:


len(data)


# In[8]:


data = np.array(data)
labels =np.array(labels)


# In[9]:


from sklearn.model_selection import train_test_split


# In[10]:


X_train,X_test,y_train,y_test = train_test_split(data,labels,test_size=0.3,random_state = 1,stratify = labels)


# In[11]:


labels.mean(), y_test.mean(), y_train.mean()


# In[12]:


X_train = X_train/255.0


# In[36]:


model = models.Sequential()
model.add(layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(224,224,3)))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(filters=32,kernel_size=2,padding='same',activation='relu'))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(layers.MaxPooling2D(pool_size=2))
model.add(layers.Dropout(0.2))

model.add(layers.Flatten())

model.add(layers.Dense(32,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(128,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(256,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(512,activation="relu"))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1024,activation="relu"))
model.add(layers.Dropout(0.2))


# In[38]:


model.add(layers.Dense(2,activation="softmax"))#5 represent output layer neurons 
model.summary()


# In[40]:


model.compile(optimizer="adam",
              loss="SparseCategoricalCrossentropy", 
             metrics=["accuracy"])  


# In[41]:


trained_model = model.fit(X_train,y_train, epochs=20, validation_data=(X_test,y_test))


# In[43]:





# In[46]:


model.save("grains_manual_k.h5")


# In[47]:


plt.plot(trained_model.history['loss'], label='train loss')
plt.plot(trained_model.history['val_loss'], label='val loss')
plt.legend()
plt.show()


# In[48]:


plt.plot(trained_model.history['accuracy'], label='train acc')
plt.plot(trained_model.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()


# In[ ]:




