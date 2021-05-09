#!/usr/bin/env python
# coding: utf-8

# OBJECTIVE 1

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def load_img():
    img = cv2.imread('../rice_images/broken_grain_2.jpg').astype(np.float32) / 255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


# In[3]:


def display_img(img):
    fig = plt.figure(figsize=(12,10))
    ax = fig.add_subplot(111)
    ax.imshow(img)


# In[4]:


i = load_img()
display_img(i)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:







# In[10]:


def number_of_grains(gray):
    ret,binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)

#averaging filter
    kernel = np.ones((5,5),np.float32)/9
    dst = cv2.filter2D(binary,-1,kernel)


    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))

#erosion
    erosion = cv2.erode(dst,kernel2,iterations = 1)

#dilation 
    dilation = cv2.dilate(erosion,kernel2,iterations = 1)

#edge detection
    edges = cv2.Canny(dilation,100,200)

### Size detection
    contours,hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print ("No. of rice grains=",len(contours))
   


# In[ ]:



   


# In[11]:


image= cv2.imread('../rice_images/broken_grain_2.jpg')

gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
number_of_grains(gray)



# In[ ]:





# OBJECTIVE 2

# In[18]:


from PIL import Image
import cv2
from keras.models import load_model
import numpy as np
import numpy as np
from keras.preprocessing import image
from keras.preprocessing import image


# In[19]:


model = load_model('grains_manual_k.h5')


# In[34]:


#file_path = "../mixed_grains_3.jpg"


# In[35]:


file_path = "../mixed_grain_1.jpg"


# In[36]:


def predict_img():
    test_image = image.load_img(file_path,target_size=(224,224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image,axis=0)
    result = model.predict(test_image)
    print("Percentage of broken grains = ", result[0,0]*100,"%")
    


# In[37]:


predict_img()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




