import streamlit as st
import cv2 
import numpy as np
import matplotlib.pyplot as plt


st.title("Task 4 - Defect Detection in Fabric")

st.text("""
    we can use two methods to detect the defects in the fabric
    1. Using the image processing techniques
    2. Using the deep learning techniques

    for the first method we can use the following steps:
        1. Convert the image to gray scale
        2. Apply Gaussian blur to the image
        3. Apply adaptive thresholding to the image
        4. Apply morphologyEx to the image
        5. Find the contours in the image
        6. Draw the contours on the image

    for the second method we can use the following steps:
        1. Convert the image to gray scale
        2. Apply Gaussian blur to the image
        3. Apply adaptive thresholding to the image
        4. get a dataset of the thresholded images because it have more details in it
        5. train the model on the dataset (YoloV8 or yolov5)
        6. detect the defects in the maskes images (thresholded images)
        7. draw the contours on the original image

    In this task we will use the first method as it needs more steps and it is more interesting and because of the few amount of images we have we cant use the second method efficiently
    and for the second method we will use the YoloV5 model and annotate the images using roboflow and the rest is easy
        """)

blr_kernel = st.slider("Blur Kernel", 1, 40, 15,step=2)
morphologyEx_kernel = st.slider("MorphologyEx Kernel", 1, 40, 1,step=2)

img_name = st.text_input("Enter image path", "Data/Task4/Fabric17.jpg")




col1, col2 = st.columns(2)
 


def enhance_fabric2(img):
    image = img.copy()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blr = cv2.GaussianBlur(gray, (blr_kernel, blr_kernel), 0)

    dst = cv2.fastNlMeansDenoising(blr, None, 10, 7, 21)

    thresh = cv2.adaptiveThreshold(dst, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # show thresholded image in streamlit
    with col2:
        st.image(thresh)
        st.write("The image is thresholded")
        
    
    
    kernal = np.ones((morphologyEx_kernel,morphologyEx_kernel), np.uint8)

    closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernal)
    
    contours,_ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        print('The image is defective')
        
        # print contours area
        area = []
        for i in contours:
            area.append(cv2.contourArea(i))
        print('The max area is: ', max(area))
        st.write('The max area is: ', max(area))

        for i in contours:
            if cv2.contourArea(i) ==  max(area):
                cv2.drawContours(image, [i], -1, (0,255,0), 3)

    else:
        print('The image is not defective')
    return image

# show the output image in streamlit

img = cv2.imread(img_name)
img = enhance_fabric2(img)
with col1:
    st.image(img)
    st.write("The image is processed")
