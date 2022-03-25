from segmenthands import *
import numpy as np
from matplotlib import pyplot as plt
import cv2.cv2 as cv2

# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/141.jpg'
img_folder_path = '../test_images/test_set/'
ensemble1_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test15'
ensemble2_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test16'
M1 = 1
M2 = 1
N_img = 2

th = 0.99

plot_img = 1

# Store combined color masks and combined probabilities
comb_col_mask1 = np.empty([N_img, 288, 384, 3])
comb_col_mask2 = np.empty([N_img, 288, 384, 3])
comb_probs1 = np.empty([N_img, 288, 384, 3])
comb_probs2 = np.empty([N_img, 288, 384, 3])

for k in range(N_img):
    img_path = img_folder_path + str(95 + k) + '.jpg'
    # Read image
    img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (384, 288))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Initialize variables
    mask1 = np.empty([M1, 288, 384])
    mask2 = np.empty([M2, 288, 384])
    probs1 = np.empty([M1, 288, 384])
    probs2 = np.empty([M2, 288, 384])
    colmask1 = np.empty([M1, 288, 384, 3])
    colmask2 = np.empty([M2, 288, 384, 3])

    for m in range(M1):
        model_path = ensemble1_path + '/model' + str(1 + m) + '.pt'
        mask1[m], probs1[m] = SegmentHands(img, model_path)

    for m in range(M2):
        model_path = ensemble2_path + '/model' + str(1 + m) + '.pt'
        mask2[m], probs2[m] = SegmentHands(img, model_path)

    # Compute combined color mask
    comb_col_mask1[k] = getcoloredMask(img, getMask(probs1, threshold=th))
    comb_col_mask2[k] = getcoloredMask(img, getMask(probs2, threshold=th))
    probs1_avg = np.average(probs1, axis=0)
    probs2_avg = np.average(probs2, axis=0)

    comb_probs1[k] = np.dstack((probs1_avg, probs1_avg, probs1_avg))
    comb_probs2[k] = np.dstack((probs2_avg, probs2_avg, probs2_avg))

if plot_img:
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(N_img, 2, 1)
    plt.imshow(np.hstack((comb_col_mask1[0], comb_probs1[0])))
    plt.axis('off')
    plt.title('One model')
    fig.add_subplot(N_img, 2, 2)
    plt.imshow(np.hstack((comb_col_mask2[0], comb_probs2[0])))
    plt.axis('off')
    plt.title('Ensemble (m=5)')
    for ii in range(N_img - 1):
        fig.add_subplot(N_img, 2, 2*ii + 3)
        plt.imshow(np.hstack((comb_col_mask1[ii+1], comb_probs1[ii+1])))
        plt.axis('off')
        fig.add_subplot(N_img, 2, 2*ii + 4)
        plt.imshow(np.hstack((comb_col_mask2[ii+1], comb_probs2[ii+1])))
        plt.axis('off')

plt.show()

