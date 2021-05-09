# Rice-grain-Image-processing
To count the number of rice grains in the image and to find the percentage of broken grains in the image.


This file consists of total three python codes. The image augmentation python code was used to create more number of image samples for CNN. The rice_custom_architecture code has the architecture for deep learning using CNN.The final code is the front end of the objective that is to find the number of rice grains in the image and to find the percentage of broken grains in the image.

Here the number of rice grains was detected with help of openCV. The cv2.findContours() of openCV was used here to find the number of rice grains by developing external contours around the image. Here before contours detection erosion is followed by dilation. Because, erosion removes white noises, but it also shrinks our object. So we dilate it. Since noise is gone, they won’t come back, but our object area increases. It is also useful in joining broken parts of an object.

![Accuracy](https://user-images.githubusercontent.com/83361041/117563962-0d1e4480-b0c7-11eb-963d-82c09bb5738c.png)
![External contour](https://user-images.githubusercontent.com/83361041/117563964-0e4f7180-b0c7-11eb-85c1-fff98a8ba6f6.png)
![Loss](https://user-images.githubusercontent.com/83361041/117563965-0ee80800-b0c7-11eb-831f-2322552d1bac.png)
![Number of rice grains](https://user-images.githubusercontent.com/83361041/117563967-0f809e80-b0c7-11eb-8922-121fe04353a5.png)
![Orginal Image](https://user-images.githubusercontent.com/83361041/117563968-10193500-b0c7-11eb-9e95-3ff6e6731665.png)
![Percentage of broken grains](https://user-images.githubusercontent.com/83361041/117563969-10193500-b0c7-11eb-8b8d-957d7d1c6f43.png)




