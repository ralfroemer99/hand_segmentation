from segmenthands import *
import numpy as np
import os
from matplotlib import pyplot as plt
from metrics import getpixelacc, getmeanIoU
import cv2.cv2 as cv2

img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/03_resized_images/test_set'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/13_resized_masks/test_set'
# Persons
p1 = np.array([83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])
p2 = np.array([74, 75, 76, 77, 78, 79, 80, 81, 82])
p3 = np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73])

# Backgrounds
b1 = np.array([51, 53, 61, 62, 63, 64, 65, 66, 68, 69, 70, 73, 81, 82])     # even
b2 = np.array([52, 54, 55, 56, 57, 58, 59, 60, 67, 71, 72, 74, 75, 76, 77, 78, 79, 80, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100])

# Environment
e1 = np.array([83, 84, 85, 86, 87, 88, 89, 90, 91, 96, 97])   # outdoors
e2 = np.array([51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 92, 93, 94, 95, 98, 99, 100])

ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'
M = 5
th = 0.5

np1 = len(p1)
np2 = len(p2)
np3 = len(p3)
nb1 = len(b1)
nb2 = len(b2)
ne1 = len(e1)
ne2 = len(e2)

# print('Total length of index lists: ' + str(n1 + n2 + n3))
print('Total length of index lists (persons): ' + str(np1 + np2 + np3))
print('Total length of index lists (background): ' + str(nb1 + nb2))
print('Total length of index lists (environment): ' + str(ne1 + ne2))

pixel_accp1 = np.array([])
mean_ioup1 = np.array([])
pixel_accp2 = np.array([])
mean_ioup2 = np.array([])
pixel_accp3 = np.array([])
mean_ioup3 = np.array([])
pixel_accb1 = np.array([])
mean_ioub1 = np.array([])
pixel_accb2 = np.array([])
mean_ioub2 = np.array([])
pixel_acce1 = np.array([])
mean_ioue1 = np.array([])
pixel_acce2 = np.array([])
mean_ioue2 = np.array([])

for index, file in enumerate(os.listdir(img_path)):
    # Read image file
    img_nr = index + 51
    input_path = img_path + '/' + str(img_nr) + '.jpg'
    img = cv2.imread(input_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize variables
    masks = np.empty([M, 288, 384])
    probs = np.empty([M, 288, 384])

    for m in range(M):
        model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
        masks[m], probs[m] = SegmentHands(img, model_path)

    # probs_avg = np.average(probs, axis=0)
    comb_mask = getMask(probs, threshold=th)

    # Read mask file
    input_path = mask_path + '/' + str(img_nr) + '.png'
    real_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)

    pixel_acc = getpixelacc(comb_mask, real_mask)
    mean_iou = getmeanIoU(comb_mask, real_mask)
    print('Image number: ' + str(img_nr) + ', pixel acc. = ' + str(np.around(pixel_acc, 3)) + ', mean IoU = ' + str(np.around(mean_iou, 3)))

    if img_nr in p1:
        pixel_accp1 = np.append(pixel_accp1, pixel_acc)
        mean_ioup1 = np.append(mean_ioup1, mean_iou)
        print('Image ' + str(img_nr) + ', person 1')
    elif img_nr in p2:
        pixel_accp2 = np.append(pixel_accp2, pixel_acc)
        mean_ioup2 = np.append(mean_ioup2, mean_iou)
        print('Image ' + str(img_nr) + ', person 2')
    else:
        pixel_accp3 = np.append(pixel_accp3, pixel_acc)
        mean_ioup3 = np.append(mean_ioup3, mean_iou)
        print('Image ' + str(img_nr) + ', person 3')

    if img_nr in b1:
        pixel_accb1 = np.append(pixel_accb1, pixel_acc)
        mean_ioub1 = np.append(mean_ioub1, mean_iou)
        print('Image ' + str(img_nr) + ', even background')
    else:
        pixel_accb2 = np.append(pixel_accb2, pixel_acc)
        mean_ioub2 = np.append(mean_ioub2, mean_iou)
        print('Image ' + str(img_nr) + ', uneven background')

    if img_nr in e1:
        pixel_acce1 = np.append(pixel_acce1, pixel_acc)
        mean_ioue1 = np.append(mean_ioue1, mean_iou)
        print('Image ' + str(img_nr) + ', outdoors')
    else:
        pixel_acce2 = np.append(pixel_acce2, pixel_acc)
        mean_ioue2 = np.append(mean_ioue2, mean_iou)
        print('Image ' + str(img_nr) + ', indoors')

print('Person 1: n_img = ' + str(len(pixel_accp1)) + ', pixel acc = ' + str(np.average(pixel_accp1)) + ', mIoU = ' + str(np.average(mean_ioup1)))
print('Person 2: n_img = ' + str(len(pixel_accp2)) + ', pixel acc = ' + str(np.average(pixel_accp2)) + ', mIoU = ' + str(np.average(mean_ioup2)))
print('Person 3: n_img = ' + str(len(pixel_accp3)) + ', pixel acc = ' + str(np.average(pixel_accp3)) + ', mIoU = ' + str(np.average(mean_ioup3)))
print('Even background: n_img = ' + str(len(pixel_accb1)) + ', pixel acc = ' + str(np.average(pixel_accb1)) + ', mIoU = ' + str(np.average(mean_ioub1)))
print('Uneven background: n_img = ' + str(len(pixel_accb2)) + ', pixel acc = ' + str(np.average(pixel_accb2)) + ', mIoU = ' + str(np.average(mean_ioub2)))
print('Outdoors: n_img = ' + str(len(pixel_acce1)) + ', pixel acc = ' + str(np.average(pixel_acce1)) + ', mIoU = ' + str(np.average(mean_ioue1)))
print('Indoors: n_img = ' + str(len(pixel_acce2)) + ', pixel acc = ' + str(np.average(pixel_acce2)) + ', mIoU = ' + str(np.average(mean_ioue2)))

