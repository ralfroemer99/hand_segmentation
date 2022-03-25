import cv2.cv2 as cv2
import numpy as np

img = cv2.imread('C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/rotated_images/1.jpg')

lower = np.array([0, 48, 80], dtype="uint8")
upper = np.array([20, 255, 255], dtype="uint8")

# Convert image to HSV color space and determine the pixel intensities falling into the specified boundaries
# img_resized = cv2.resize(img, ())
converted = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(converted, lower, upper)

# Apply a series of erosions and dilations to the mask using an elliptical kernel
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
mask = cv2.erode(mask, kernel, iterations=2)
mask = cv2.dilate(mask, kernel, iterations=2)

# Blur the mask to remove Gaussian noise, then apply the mask to the frame
mask = cv2.GaussianBlur(mask, (3, 3), 0)
skin = cv2.bitwise_and(img, img, mask=mask)

# Show original image and skin
cv2.imshow("test_images", cv2.resize(np.hstack([img, skin]), (800, 300)))

cv2.imwrite('C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/masks/1.png', mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
