import cv2
from segmenthands import *
import numpy as np
from numpy import savetxt, loadtxt
import matplotlib.pyplot as plt
import tikzplotlib
from metrics import getpixelacc, accoverconf
from segmenthands import getMask

img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/images_resized'
# img_path = 'test_images/test3.png'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/hand_over_face/masks'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test3'

M = 5
N = 25
generate_data = 0

if generate_data:
    # Initialize pixel accuracy matrix
    acc = np.empty((M+1, N))

    for n in range(N):
        print('n = ' + str(n))

        # Determine name of next image
        data = n + 200

        # Read image
        img = cv2.imread(img_path + '/' + str(data) + '.jpg', cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img, (384, 288))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Read mask
        real_mask = cv2.imread(mask_path + '/' + str(data) + '.png', cv2.IMREAD_GRAYSCALE) / 255
        real_mask = cv2.resize(real_mask, (384, 288))

        # Initialize variables
        mask = np.empty([288, 384, M])
        probs = np.empty([288, 384, M])
        for m in range(M):
            model_path = ensemble_path + '/model' + str(m + 1) + '.pt'
            mask[:, :, m], probs[:, :, m] = SegmentHands(img, model_path)
            acc[m, n] = getpixelacc(mask[:, :, m], real_mask)

        comb_mask = getMask(np.average(probs, axis=2))
        acc[M, n] = getpixelacc(comb_mask, real_mask)
    savetxt('data/pixel_acc2.csv', acc, delimiter=',')


# Plot
acc = loadtxt('data/pixel_acc2.csv', delimiter=',')
# acc = np.delete(acc, 16, 1)
n_vec = np.linspace(1, N, N)
for m in range(M):
    plt.plot(n_vec, acc[m, :], 'b--')
    print(np.average(acc[m, :]))

print(np.average(acc[M, :]))
plt.plot(n_vec, acc[M, :], 'r-')
tikzplotlib.save('C:/Users/ralf-/OneDrive/!Uni/Seminar/Presentation/slides-tex/pics/pixel_acc3.tex')
plt.show()

