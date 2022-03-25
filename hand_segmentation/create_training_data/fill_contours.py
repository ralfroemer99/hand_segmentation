import cv2.cv2 as cv2
import numpy as np
import os

path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/11_drawn_contours/bla'
outPath = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/12_masks/bla'


# Bounds for the green pixel detection
lower = np.array([0, 205, 10], dtype="uint8")
upper = np.array([20, 255, 66], dtype="uint8")

for index, file in enumerate(os.listdir(path)):
    # create the full input path and read the file
    input_path = path + '/' + str(index + 71) + '.jpg'
    img = cv2.imread(input_path)

    # Extract self-drawn green contour image
    contour = cv2.inRange(img, lower, upper)

    # Find and fill contour
    cnts = cv2.findContours(contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    for c in cnts:
        cv2.drawContours(contour, [c], 0, (255, 255, 255), -1)

    cv2.imwrite(outPath + '/' + str(index + 1) + '.png', contour)

