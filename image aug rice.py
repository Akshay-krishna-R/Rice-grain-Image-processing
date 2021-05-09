#!/usr/bin/env python
# coding: utf-8

# In[18]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


full_n = cv2.imread('./rice_images/full_grain_7.jpg')
full_n = cv2.cvtColor(full_n, cv2.COLOR_BGR2RGB)


# In[44]:


plt.imshow(full_n)


# In[45]:


from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='constant')

img = load_img('./rice_images/full_grain_7.jpg')  # this is a PIL image
x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)

# the .flow() command below generates batches of randomly transformed images
# and saves the results to the `preview/` directory
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='./image aug/full_grains', save_prefix='full_grains', save_format='jpeg'):
    i += 1
    if i > 101:
        break


# In[ ]:




