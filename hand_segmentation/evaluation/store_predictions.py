from segmenthands import *
import numpy as np
import os
from matplotlib import pyplot as plt
from metrics import getpixelacc, acc_over_conf, getmeanIoU
import cv2.cv2 as cv2

# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/141.jpg'
img_path = '../test_images/test_set/'
mask_path = '../test_images/mask_set/'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'
img_out_path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/Evaluation/col_masks_th01'
probs_out_path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/Evaluation/probs_bad'
M = 5
N_img = 50
th = 0.1


for index, file in enumerate(os.listdir(img_path)):
    img_nr = index + 51
    print('Image number: ' + str(img_nr))
    # Read image file
    input_path = img_path + '/' + str(img_nr) + '.jpg'
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize variables
    masks = np.empty([M, 288, 384])
    probs = np.empty([M, 288, 384])

    for m in range(M):
        model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
        masks[m], probs[m] = SegmentHands(img, model_path)

    # Compute average probabilities
    probs_avg = np.average(probs, axis=0)*255

    # Compute color mask
    comb_mask = getMask(probs, threshold=th)
    col_mask = getcoloredMask(img, comb_mask)*255
    col_mask = np.array(col_mask, np.uint8)
    col_mask = cv2.cvtColor(col_mask, cv2.COLOR_RGB2BGR)

    # Get masks for storing color mask and probabilities
    out_path = img_out_path + '/' + str(img_nr) + '.jpg'
    cv2.imwrite(out_path, col_mask)

    # out_path = probs_out_path + '/' + str(img_nr) + '.png'
    # cv2.imwrite(out_path, probs_avg)