# can pass np array or path to image file
import numpy as np
import torch
import cv2
from torchvision.transforms import transforms

from fetchmodel import HandSegModel
import PIL.Image as Image


def getMask(probs, threshold=None):
    if threshold is None:
        threshold = 0.5

    if probs.ndim == 2:
        comb_mask = probs >= threshold
    else:
        comb_mask = (np.average(probs, axis=0) >= threshold)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # comb_mask = cv2.dilate(comb_mask.astype('float'), kernel, iterations=2)
    comb_mask = cv2.morphologyEx(comb_mask.astype('float'), cv2.MORPH_CLOSE, kernel)
    comb_mask = cv2.morphologyEx(comb_mask, cv2.MORPH_OPEN, kernel)

    return comb_mask


def SegmentHands(img_path, model_path=None, threshold=None):
    if isinstance(img_path, np.ndarray):
        img = Image.fromarray(img_path)
    else:
        img = Image.open(img_path)

    # Transform image to pixel range [0, 1]
    preprocess = transforms.Compose([transforms.Resize((288, 384), 2),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    Xtest = preprocess(img)

    if model_path is not None:
        checkpoint = torch.load(model_path)
    else:
        checkpoint = torch.load('C:/Users/ralf-/Documents/Python/SemanticSegmentation/checkpoints/checkpointhandseg3.pt')

    model = HandSegModel()
    model.load_state_dict(checkpoint['state_dict'])
    with torch.no_grad():
        model.eval()
        if torch.cuda.is_available():
            dev = "cuda:0"
        else:
            dev = "cpu"
        device = torch.device(dev)
        model.to(device)
        Xtest = Xtest.to(device).float()
        Xtest = Xtest.unsqueeze(0).float()      # Add extra batch dimension to make image a 4D tensor as expected by the model
        ytest = model(Xtest)                    # Forward propagate image tensor through the model
        # ypos = ytest[0, 1, :, :].clone().detach().cpu().numpy()
        # yneg = ytest[0, 0, :, :].clone().detach().cpu().numpy()
        # ymask = ypos >= yneg
        yprobs = ytest[0].detach().cpu().numpy()
        probs = np.exp(yprobs[1, :, :])/(np.sum(np.exp(yprobs), axis=0))    # softmax output function

    mask = getMask(probs, threshold)

    return mask, probs


def getcoloredMask(image, mask):
    color_mask = np.zeros_like(image, dtype=float)
    color_mask[:, :, 1] += mask * 0.8
    # color_mask[:, :, 1] += mask * 0.5
    image = image.astype(float)/255
    masked = cv2.addWeighted(image, 1.0, color_mask, 1.0, 0.0)
    masked[:, :, 1] = np.minimum(masked[:, :, 1], 1.0)
    return masked


def getUncertainty(probs):
    return np.maximum(probs, 1.0 - probs)


def getEnsemblePrediction(img_path, ensemble_path, M):
    mask = np.empty([288, 384, M])
    probs = np.empty([288, 384, M])
    for m in range(M):
        model_path = ensemble_path + '/model' + str(m + 1) + '.pt'
        mask[:, :, m], probs[:, :, m] = SegmentHands(img_path, model_path)

    comb_mask = np.average(mask, axis=2)
    comb_probs = np.average(probs, axis=2)

    return comb_mask, comb_probs