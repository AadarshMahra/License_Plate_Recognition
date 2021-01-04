import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img1 = cv2.imread('media/bmw.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
# plt.imshow(cv2.cvtColor(img1_gray, cv2.COLOR_BGR2RGB))
# plt.show()
b_filter = cv2.bilateralFilter(img1_gray, 11, 17, 17)
edged = cv2.Canny(b_filter, 30, 200)
# plt.imshow(cv2.cvtColor(edged, cv2.COLOR_BGR2RGB))
# plt.show()

keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break
# print(location)
# print(location.flatten())
# print(location[1][0], location[3][0])
# print(tuple(location[0][0]))
# print(tuple(location[2][0]))
# PRINT RECTANGLE AROUND LICENSE PLATE
detected = cv2.rectangle(img1, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 2)
# cv2.imshow('detected image', detected)
# cv2.waitKey(0)
plt.imshow(cv2.cvtColor(detected, cv2.COLOR_BGR2RGB))
plt.show()
# PRINT CROPPED LICENSE PLATE
(x1, y1) = tuple(location[0][0])
(x2, y2) = tuple(location[2][0])
cropped = img1_gray[y1:y2+1, x1:x2+1]
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.show()
# USE EASYOCR TO READ TEXT
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped)
print(result)
print("RESULT: {}".format(result[0][1]))
