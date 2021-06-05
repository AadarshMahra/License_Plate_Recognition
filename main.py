import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

img1 = cv2.imread('media/bmw.jpg')
img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
b_filter = cv2.bilateralFilter(img1_gray, 11, 17, 17)  # noise removal
edges = cv2.Canny(b_filter, 30, 200)  # locates edges using Canny algorithm
contours = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# ^ arguments: 1. source image, 2. contour retrieval mode, 3. contour approx. method

contours = imutils.grab_contours(contours)
# contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
location = None
for contour in contours:
    approx = cv2.approxPolyDP(contour, 10, True)
    if len(approx) == 4:  # if the approximated polygon has four vertices,
        location = approx  # save its vertices' coordinates
        break

# PRINT RECTANGLE AROUND LICENSE PLATE
detect_rect = cv2.rectangle(img1, tuple(location[0][0]), tuple(location[2][0]), (0, 255, 0), 4)
# ^ arguments: 1. source image, 2. top left coordinate, 3. bottom right coordinate, 4. RGB, 5. thickness
# ^^ creates a new image where rectangle is imposed on top of original car image
plt.imshow(cv2.cvtColor(detect_rect, cv2.COLOR_BGR2RGB))  # convert to RGB before plt.show()
plt.show()  # display imposed rectangle
# PRINT CROPPED LICENSE PLATE
(x1, y1) = tuple(location[0][0])
(x2, y2) = tuple(location[2][0])
cropped = img1_gray[y1:y2+1, x1:x2+1]
plt.imshow(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
plt.show()  # display cropped license plate

# USE EASYOCR TO READ TEXT
reader = easyocr.Reader(['en'])
result = reader.readtext(cropped)
print("RESULT: {}".format(result[0][1]))
