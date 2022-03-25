from segmenthands import *
import numpy as np
import os
from matplotlib import pyplot as plt
from metrics import getpixelacc, acc_over_conf, getmeanIoU
import cv2.cv2 as cv2

# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/141.jpg'
img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/images/test_set'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/masks/test_set'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/models/11'
M = 1
N_img = 50
th = np.linspace(0.5, 0.5, 1)
n_th = len(th)
# th = 0.5
# n_th = 1

avg_pixel_acc = np.empty([n_th])
avg_mean_iou = np.empty([n_th])

for k in range(n_th):
    print('New threshold: th = ' + str(th[k]))
    pixel_acc = np.empty([N_img])
    mean_iou = np.empty([N_img])
    for index, file in enumerate(os.listdir(img_path)):
        # Read image file
        input_path = img_path + '/' + str(index + 51) + '.jpg'
        img = cv2.imread(input_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Initialize variables
        masks = np.empty([M, 288, 384])
        probs = np.empty([M, 288, 384])

        for m in range(M):
            model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
            masks[m], probs[m] = SegmentHands(img, model_path)

        # probs_avg = np.average(probs, axis=0)
        comb_mask = getMask(probs, threshold=th[k])

        # Read mask file
        input_path = mask_path + '/' + str(index + 51) + '.png'
        real_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

        pixel_acc[index] = getpixelacc(comb_mask, real_mask)
        mean_iou[index] = getmeanIoU(comb_mask, real_mask)
        print('Image number: ' + str(index + 51) + ', threshold: ' + str(th[k]) + ', pixel acc. = ' + str(np.around(pixel_acc[index], 3)) + ', mean IoU = ' + str(np.around(mean_iou[index], 3)))

    avg_pixel_acc[k] = np.average(pixel_acc)
    avg_mean_iou[k] = np.average(mean_iou)
    pixel_acc_std = np.std(pixel_acc)
    mean_iou_std = np.std(mean_iou)
    print('Threshold: ' + str(th[k]) + ', avg pixel acc = ' + str(np.around(avg_pixel_acc[k], 4)) + ' +- ' + str(np.around(pixel_acc_std, 4)) + \
          ', avg mean iou = ' + str(np.around(avg_mean_iou[k], 4)) + ' +- ' + str(np.around(mean_iou_std, 4)))

print(avg_pixel_acc)
print(avg_mean_iou)
