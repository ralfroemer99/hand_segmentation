import cv2
import cv2.cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from create_training_data.multiply_data import add_noise
from albumentations import GridDistortion

bla = np.array([[1, 2], [3, 4]])
print(bla[1])


img = cv.imread('C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/03_resized_images/23.jpg', cv.IMREAD_UNCHANGED)
# img = cv.resize(img, (384, 288))
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = np.array(img, np.uint8)
mask = cv.imread('C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/13_resized_masks/23.png')

img_rgb_noise = add_noise(img, 50)

img_hsv = cv.cvtColor(img, cv.COLOR_RGB2HSV)
img_hsv_noise = add_noise(img_hsv, 50)
img_hsv_noise = np.array(img_hsv_noise, np.uint8)
img_hsv_noise = cv.cvtColor(img_hsv_noise, cv.COLOR_HSV2RGB)

aug = GridDistortion(interpolation=cv.INTER_NEAREST)
augmented = aug(image=img, mask=mask)
img_dist1 = augmented['image']
mask_dist1 = augmented['mask']
aug.border_mode = cv2.BORDER_CONSTANT
augmented = aug(image=img, mask=mask)
img_dist2 = augmented['image']
mask_dist2 = augmented['mask']
aug.border_mode = cv2.BORDER_REPLICATE
augmented = aug(image=img, mask=mask)
img_dist3 = augmented['image']
mask_dist3 = augmented['mask']
aug.border_mode = cv2.BORDER_WRAP
augmented = aug(image=img, mask=mask)
img_dist4 = augmented['image']
mask_dist4 = augmented['mask']

# plt.imshow(cv.cvtColor(np.hstack((img, img_noise)), cv.COLOR_RGB2BGR))
# plt.imshow(np.hstack((img, img_rgb_noise, img_hsv_noise, img_dist)))
# plt.show()
fig = plt.figure(figsize=(10, 5))
fig.add_subplot(2, 5, 1)
plt.imshow(img)
plt.axis('off')
plt.title('Original')
fig.add_subplot(2, 5, 6)
plt.imshow(mask, cmap='gray')
plt.axis('off')
fig.add_subplot(2, 5, 2)
plt.imshow(img_dist1)
plt.axis('off')
plt.title('BORDER_REFLECT_101')
fig.add_subplot(2, 5, 7)
plt.imshow(mask_dist1)
plt.axis('off')
fig.add_subplot(2, 5, 3)
plt.imshow(img_dist2)
plt.axis('off')
plt.title('BORDER_REFLECT_CONSTANT')
fig.add_subplot(2, 5, 8)
plt.imshow(mask_dist2)
plt.axis('off')
fig.add_subplot(2, 5, 4)
plt.imshow(img_dist3)
plt.axis('off')
plt.title('BORDER_REPLICATE')
fig.add_subplot(2, 5, 9)
plt.imshow(mask_dist3)
plt.axis('off')
fig.add_subplot(2, 5, 5)
plt.imshow(img_dist4)
plt.axis('off')
plt.title('BORDER_WRAP')
fig.add_subplot(2, 5, 10)
plt.imshow(mask_dist4)
plt.axis('off')
plt.show()
