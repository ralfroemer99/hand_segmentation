from segmenthands import *
import numpy as np
import os
from matplotlib import pyplot as plt
from metrics import getpixelacc, acc_over_conf, getmeanIoU, conf_correct
import cv2.cv2 as cv2

# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/141.jpg'
img_path = '../test_images/test_set/'
mask_path = '../test_images/mask_set/'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'
M = 5
N_img = 50
th = 0.5

probs_all = np.empty([N_img, 288, 384])
probsm_all = np.empty([M, N_img, 288, 384])
real_masks_all = np.empty([N_img, 288, 384])

for k, file in enumerate(os.listdir(img_path)):
    img_nr = k + 51
    print('Image number: ' + str(img_nr))
    # Read image file
    input_path = img_path + '/' + str(img_nr) + '.jpg'
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read mask file
    input_path = mask_path + '/' + str(img_nr) + '.png'
    real_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    # Initialize variables
    masks = np.empty([M, 288, 384])
    probs = np.empty([M, 288, 384])

    for m in range(M):
        model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
        masks[m], probs[m] = SegmentHands(img, model_path)
        probsm_all[m, k] = probs[m]

    probs_all[k] = np.average(probs, axis=0)
    real_masks_all[k] = real_mask

conf_corr, conf_uncorr = conf_correct(probs_all, real_masks_all)
conf_corr1, conf_uncorr1 = conf_correct(probsm_all[0], real_masks_all)
conf_corr2, conf_uncorr2 = conf_correct(probsm_all[1], real_masks_all)
conf_corr3, conf_uncorr3 = conf_correct(probsm_all[2], real_masks_all)
conf_corr4, conf_uncorr4 = conf_correct(probsm_all[3], real_masks_all)
conf_corr5, conf_uncorr5 = conf_correct(probsm_all[4], real_masks_all)
print('Ensemble: unc_corr = ' + str(conf_corr) + ', unc_uncorr = ' + str(conf_uncorr))
print('m = 1: unc_corr = ' + str(conf_corr1) + ', unc_uncorr = ' + str(conf_uncorr1))
print('m = 2: unc_corr = ' + str(conf_corr2) + ', unc_uncorr = ' + str(conf_uncorr2))
print('m = 3: unc_corr = ' + str(conf_corr3) + ', unc_uncorr = ' + str(conf_uncorr3))
print('m = 4: unc_corr = ' + str(conf_corr4) + ', unc_uncorr = ' + str(conf_uncorr4))
print('m = 5: unc_corr = ' + str(conf_corr5) + ', unc_uncorr = ' + str(conf_uncorr5))
