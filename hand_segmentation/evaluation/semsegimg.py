import cv2
from segmenthands import *
import numpy as np
import matplotlib.pyplot as plt
from metrics import getpixelacc, accoverconf

# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/20.jpg'
img_path = 'test_images/test3.png'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/masks/20.png'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'

M = 5

# Read image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (384, 288))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Read real mask
# real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
# real_mask = cv2.resize(real_mask, (384, 288))

# Initialize variables
mask = np.empty([288, 384, M])
probs = np.empty([288, 384, M])
unc = np.empty([288, 384, M])
colmask = np.empty([288, 384, 3, M])

model_path = ensemble_path + '/model' + str(1 + 1) + '.pt'
mask, probs = SegmentHands(img, model_path)
# print('Pixel accuracy = ' + str(np.around(getpixelacc(mask, real_mask), 3)))

cv2.imwrite('C:/Users/ralf-/OneDrive/!Uni/Seminar/Presentation/slides-tex/pics/trump_mask.jpg', cv2.resize(mask, (800, 533))*255)

