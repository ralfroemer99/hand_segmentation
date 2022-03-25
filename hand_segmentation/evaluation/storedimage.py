import cv2
from segmenthands import *
import numpy as np
import matplotlib
# matplotlib.use("pgf")
# matplotlib.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False,
# })
from matplotlib import pyplot as plt
from metrics import getpixelacc

img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized/141.jpg'
# img_path = 'test_images/test3.png'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/masks/141.png'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test17'
plot_img = 1

M = 5

# Read image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (384, 288))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# Read real mask
real_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)/255
real_mask = cv2.resize(real_mask, (384, 288))

# Initialize variables
mask = np.empty([288, 384, M])
probs = np.empty([288, 384, M])
unc = np.empty([288, 384, M])
colmask = np.empty([288, 384, 3, M])

for m in range(M):
    model_path = ensemble_path + '/model' + str(1 + m) + '.pt'
    mask[:, :, m], probs[:, :, m] = SegmentHands(img, model_path)
    print('m = ' + str(m) + ': pixel accuracy = ' + str(np.around(getpixelacc(mask[:, :, m], real_mask), 3)))
    if plot_img:
        unc[:, :, m] = getUncertainty(probs[:, :, m])
        colmask[:, :, :, m] = getcoloredMask(img, mask[:, :, m])

# Compute combined mask
comb_mask = getMask(probs)
comb_col_mask = getcoloredMask(img, comb_mask)
print('Ensemble: pixel accuracy = ' + str(np.around(getpixelacc(comb_mask, real_mask), 3)))

# conf, acc = accoverconf(np.average(probs, axis=2), real_mask)

# Plot accuracy over confidence
# if acc.shape[1] == 1:
#     plt.plot(conf, acc)
#     plt.title('Accuracy over confidence of the ensemble')
# else:
#     plt.plot(conf, acc[:, -1], 'r-')
#     for m in range(acc.shape[1]-1):
#         plt.plot(conf, acc[:, m], 'b--')
#     plt.title('Accuracy over confidence of the models and the ensemble')
#
# plt.xlabel('Confidence')
# plt.ylabel('Accuracy')
# plt.show

if plot_img:
    fig = plt.figure(figsize=(10, 5))
    for m in range(M):
        fig.add_subplot(3, M + 1, m+1)
        plt.imshow(colmask[:, :, :, m])
        plt.axis('off')
        plt.title('m = ' + str(m))
        plt.imsave('results/ensemble_img' + str(m*3) + '.jpg', colmask[:, :, :, m])
        fig.add_subplot(3, M + 1, m+M+2)
        plt.imshow(probs[:, :, m], cmap='gray')
        plt.axis('off')
        plt.imsave('results/ensemble_img' + str(m * 3 + 1) + '.jpg', probs[:, :, m], cmap='gray')
        # plt.title('m = ' + str(m))
        fig.add_subplot(3, M + 1, m+2*M+3)
        plt.imshow(unc[:, :, m], cmap='gray')
        plt.axis('off')
        plt.imsave('results/ensemble_img' + str(m * 3 + 2) + '.jpg', unc[:, :, m], cmap='gray')
        # plt.title('m = ' + str(m))

    fig.add_subplot(3, M + 1, M+1)
    plt.imshow(comb_col_mask)
    plt.axis('off')
    plt.title('Ensemble')
    plt.imsave('results/ensemble_img' + str(M * 3) + '.jpg', comb_col_mask)
    fig.add_subplot(3, M + 1, 2*(M+1))
    plt.imshow(np.average(probs, axis=2), cmap='gray')
    plt.axis('off')
    plt.imsave('results/ensemble_img' + str(M * 3 + 1) + '.jpg', np.average(probs, axis=2), cmap='gray')
    # plt.title('Ensemble')
    fig.add_subplot(3, M + 1, 3*(M+1))
    plt.imshow(getUncertainty(np.average(probs, axis=2)), cmap='gray')
    plt.axis('off')
    plt.imsave('results/ensemble_img' + str(M * 3 + 2) + '.jpg', getUncertainty(np.average(probs, axis=2)), cmap='gray')
    # plt.title('Ensemble')
    # plt.savefig('results/ensemble.pgf')

    plt.show()

