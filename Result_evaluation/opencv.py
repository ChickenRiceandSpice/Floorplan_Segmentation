import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np



img_path = "/home/gen79net/Downloads/Image Detection/Result_evaluation/Pgm_Images/image28.pgm"

img = cv.imread(img_path)[:,:,1]
img_1 = cv.imread(img_path)[:,:,1]
# plt.imshow(img)
# plt.show()

hist = cv.calcHist([img],[0],None ,[256],[0,255])
# plt.hist(img.ravel(),256, [0,255])
# plt.show()

thresh_img = cv.adaptiveThreshold(img,10,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY, 5,2)
# plt.imshow(thresh_img)
# plt.show()

contours, hierarchy = cv.findContours(thresh_img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(thresh_img, contours, -1, (0,255),1)

dilate_kernel = cv.getStructuringElement(cv.MORPH_RECT,(2,3))
dilate = cv.dilate(thresh_img, dilate_kernel, iterations = 2)


close_kernel = cv.getStructuringElement(cv.MORPH_RECT,(1,3))
close_img = cv.morphologyEx(dilate, cv.MORPH_CLOSE, close_kernel, iterations = 1)


fig, ax = plt.subplots(1,2)
ax[0].imshow(close_img)
ax[1].imshow(img_1)
plt.show()

