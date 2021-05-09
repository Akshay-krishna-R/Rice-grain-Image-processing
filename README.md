# Rice-grain-Image-processing
To count the number of rice grains in the image and to find the percentage of broken grains in the image.


This file consists of total three python codes. The image augmentation python code was used to create more number of image samples for CNN. The rice_custom_architecture code has the architecture for deep learning using CNN.The final code is the front end of the objective that is to find the number of rice grains in the image and to find the percentage of broken grains in the image.

Here the number of rice grains was detected with help of openCV. The cv2.findContours() of openCV was used here to find the number of rice grains by developing external contours around the image. Here before contours detection erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, but our object area increases. It is also useful in joining broken parts of an object.

![Accuracy](https://user-images.githubusercontent.com/83361041/117564074-ca10a100-b0c7-11eb-9cf0-da62f1c6b879.png)
Accuracy of model

![External contour](https://user-images.githubusercontent.com/83361041/117564076-cb41ce00-b0c7-11eb-8798-c81079604122.png)
External Contour 

![Loss](https://user-images.githubusercontent.com/83361041/117564077-cbda6480-b0c7-11eb-9232-7694ea1cfa6e.png)
Loss of model 

![Number of rice grains](https://user-images.githubusercontent.com/83361041/117564078-cbda6480-b0c7-11eb-8bd6-b2dd07189599.png)
Number of rice grains

![Orginal Image](https://user-images.githubusercontent.com/83361041/117564080-cc72fb00-b0c7-11eb-8586-77904162bc9b.png)
Original Image

![Percentage of broken grains](https://user-images.githubusercontent.com/83361041/117564081-cd0b9180-b0c7-11eb-8315-4dc3b52cf9c2.png)
Percentage of broken grain





