import numpy as np
from segmenthands import *
import os
import cv2.cv2 as cv
# from create_training_data.multiply_data import add_noise, change_background
from albumentations import GridDistortion

from segmenthands import getMask
col_mask_path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/Evaluation/col_masks_th01'
probs_path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/Evaluation/probs_bad'

col_mask_full = np.empty((288*10, 384*5, 3))
probs_full = np.empty((288*10, 384*5, 3))

N_img = 50

for k in range(N_img):
    input_path = col_mask_path + '/' + str(k+51) + '.jpg'
    col_mask = cv2.imread(input_path)

    input_path = probs_path + '/' + str(k + 51) + '.png'
    probs = cv2.imread(input_path)

    col = int(np.mod(k, 5))
    row = int(np.floor(k / 5))
    col_mask_full[288 * row:288 * (row + 1), 384 * col:384 * (col + 1), :] = col_mask
    probs_full[288 * row:288 * (row + 1), 384 * col:384 * (col + 1), :] = probs

cv.imwrite('C:/Users/ralf-/OneDrive/!Uni/Seminar/Presentation/Final presentation/pics/all_col_masks_th01.png', col_mask_full)
# cv.imwrite('C:/Users/ralf-/OneDrive/!Uni/Seminar/Presentation/Final presentation/pics/all_probs_bad.png', probs_full)

