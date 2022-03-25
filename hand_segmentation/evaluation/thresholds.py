import matplotlib.pyplot as plt
from segmenthands import *
import cv2

# Specify paths
img_path = '/test_images/test5.jpg'
ensemble_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/ensembles/test3'

# Number of models in the ensemble
M = 5

# Threshold probabilities
th = np.array([0.5, 0.3, 0.1])

# Read image
img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = cv2.resize(img, (384, 288))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Get ensemble prediction
_, probs = getEnsemblePrediction(img_path, ensemble_path, M)

mask = np.ndarray([288, 384, th.size])
colmask = np.ndarray([288, 384, 3, th.size])

for k in range(th.size):
    mask[:, :, k] = getMask(probs, th[k])
    colmask[:, :, :, k] = getcoloredMask(img, mask[:, :, k])


# Plot results
fig = plt.figure(figsize=(10, 5))
for k in range(th.size):
    fig.add_subplot(1, th.size+1, k+1)
    plt.imshow(colmask[:, :, :, k])
    plt.axis('off')
    plt.title('threshold = ' + str(th[k]))
    plt.imsave('C:/Users/ralf-/OneDrive/!Uni/Seminar/hand_segmentation/results/threshold_mask' + str(k) + '.jpg', colmask[:, :, :, k])

plt.show()

