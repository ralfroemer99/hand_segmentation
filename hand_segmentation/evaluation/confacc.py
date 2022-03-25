from segmenthands import *
import numpy as np
import os
from matplotlib import pyplot as plt
from metrics import getpixelacc, acc_over_conf, getmeanIoU
import cv2.cv2 as cv2

img_path = '../test_images/test_set/'
mask_path = '../test_images/mask_set/'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'

M = 5
N_img = 50
th = 0.5

probs_all = np.empty((N_img, 288, 384))
probsm_all = np.empty((M, N_img, 288, 384))
masks_all = np.empty((N_img, 288, 384))

for index, file in enumerate(os.listdir(img_path)):
    img_nr = index + 51
    print('Image number: ' + str(img_nr))
    # Read image file
    input_path = img_path + '/' + str(img_nr) + '.jpg'
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    probs = np.empty([M, 288, 384])

    for m in range(M):
        model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
        _, probs[m] = SegmentHands(img, model_path)
        probsm_all[m, index, :, :] = probs[m]

    input_path = mask_path + '/' + str(img_nr) + '.png'
    mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    mask = (mask >= 127)

    # Add predicted probabilities and real mask to array
    probs_all[index] = np.average(probs, axis=0)
    masks_all[index] = mask


conf_th, acc = acc_over_conf(probs_all, masks_all)
_, acc1 = acc_over_conf(probsm_all[0], masks_all)
_, acc2 = acc_over_conf(probsm_all[1], masks_all)
_, acc3 = acc_over_conf(probsm_all[2], masks_all)
_, acc4 = acc_over_conf(probsm_all[3], masks_all)
_, acc5 = acc_over_conf(probsm_all[4], masks_all)
print(conf_th)
print(acc)
print(acc1)
print(acc2)
print(acc3)
print(acc4)
print(acc5)

