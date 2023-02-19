import streamlit as st


# # Import libraries
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
  
# image = cv2.imread('Data/task3/freezer_image (1).jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  
# blur = cv2.GaussianBlur(gray, (21, 21), 0)
# canny = cv2.Canny(blur, 30, 150, 3)
# dilated = cv2.dilate(canny, (1, 1), iterations=0)
  
# (cnt, hierarchy) = cv2.findContours(
#     dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
# rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)
  
# plt.imshow(rgb)
# plt.show()
# print("coins in the image : ", len(cnt))


st.title("task3-counting objects in an image")

img = st.text_input("Enter image path", "Data/task3/freezer_image (1).jpg")
conf = st.slider("confidence", 1, 100, 20,step=1)
overlap = st.slider("overlap", 1, 100, 30,step=1)

def yolov5_methode(img,conf,overlap):


    from roboflow import Roboflow
    rf = Roboflow(api_key="cjq2nO52r7xaxlB7tzMF")
    project = rf.workspace().project("retail-objects-sku110k")
    model = project.version(1).model

    # infer on a local image

    # visualize your prediction
    # model.predict("Data/task3/freezer_image (1).jpg", confidence=20, overlap=30).save("prediction2.jpg")
    pred = model.predict(img, confidence=20, overlap=30).json()
    #show image
    st.image('prediction2.jpg')
    #count the number of objects
    st.write(f'num of objects: {len(pred["predictions"])}')


yolov5_methode(img,conf,overlap)


# I could've used an image prossecing technique to count the number of objects in the image but I used a deep learning model to do that because it's more accurate