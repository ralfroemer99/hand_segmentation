import numpy as np
from segmenthands import getMask


def getmeanIoU(pred_masks, real_masks):
    if pred_masks.shape != real_masks.shape:
        print('Inputs must be of equal dimension!')
        return

    if len(pred_masks.shape) == 2:
        pred_masks = pred_masks[np.newaxis, :, :]
        real_masks = real_masks[np.newaxis, :, :]

    n = pred_masks.shape[0]      # Change to [0]
    # Change entries to boolean
    pred_masks = (pred_masks != 0)
    real_masks = (real_masks != 0)
    iou_sum = 0.0
    for ii in range(n):
        iou_score = 0.0
        # Compute IoU for semantic class 0
        intersection = np.logical_and(pred_masks[ii], real_masks[ii]).sum()
        union = np.logical_or(pred_masks[ii], real_masks[ii]).sum()
        if union == 0:
            iou_score += 0.0
        else:
            iou_score += intersection / union

        # Compute IoU for semantic class 1
        intersection = np.logical_and(1-pred_masks[ii], 1-real_masks[ii]).sum()
        union = np.logical_or(1-pred_masks[ii], 1-real_masks[ii]).sum()
        if union == 0:
            iou_score += 0
        else:
            iou_score += intersection / union

        iou_sum += iou_score / 2

    return iou_sum / n


def getpixelacc(pred_masks, real_masks):
    if pred_masks.shape != real_masks.shape:
        print('Inputs must be of equal dimension!')
        return

    pred_masks = (pred_masks != 0)
    real_masks = (real_masks != 0)

    if len(pred_masks.shape) == 2:
        pred_masks = pred_masks[np.newaxis, :, :]
        real_masks = real_masks[np.newaxis, :, :]

    # Get total number of masks
    n = pred_masks.shape[0]  # Change to [0]

    # Get total number of pixels
    a, b = pred_masks[0].shape
    n_total = a * b

    acc = 0.0
    for ii in range(n):
        n_corr = np.equal(pred_masks, real_masks).sum()
        acc += n_corr/n_total

    return acc / n


def meanIOU(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    iousum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        intersection = np.logical_and(target_arr, predicted_arr).sum()
        union = np.logical_or(target_arr, predicted_arr).sum()
        if union == 0:
            iou_score = 0
        else:
            iou_score = intersection / union
        iousum += iou_score

    miou = iousum / target.shape[0]
    return miou


def pixelAcc(target, predicted):
    if target.shape != predicted.shape:
        print("target has dimension", target.shape, ", predicted values have shape", predicted.shape)
        return

    if target.dim() != 4:
        print("target has dim", target.dim(), ", Must be 4.")
        return

    accsum = 0
    for i in range(target.shape[0]):
        target_arr = target[i, :, :, :].clone().detach().cpu().numpy().argmax(0)
        predicted_arr = predicted[i, :, :, :].clone().detach().cpu().numpy().argmax(0)

        same = (target_arr == predicted_arr).sum()
        a, b = target_arr.shape
        total = a * b
        accsum += same / total

    pixelAccuracy = accsum / target.shape[0]
    return pixelAccuracy


def getNLL(probs, real_mask):
    a, b = real_mask.shape
    N = a * b
    return 1/N * (np.multiply(real_mask, np.log(probs)) + np.multiply(1.0 - real_mask, np.log(1 - probs)))


def acc_over_conf(p, real_mask):
    # Check if multiple images and masks are provided
    if p.ndim == 2:    # Only one image
        p = p[np.newaxis, ...]
        real_mask = real_mask[np.newaxis, ...]

    # Get predicted segmentation mask
    pred_mask = (p >= 0.5)
    conf = np.maximum(p, 1.0 - p)

    # Define confidence thresholds
    n_th = 10
    th = np.linspace(0.5, 0.95, 10)
    acc = np.empty([n_th])

    equiv_total = (pred_mask == real_mask)
    n_equiv_total = equiv_total.sum()

    for ii in range(n_th):
        considered = (conf >= th[ii])
        equiv_considered = np.multiply(considered, equiv_total)
        tmp1 = equiv_considered.sum()
        tmp2 = considered.sum()
        if tmp2 != 0:
            acc[ii] = tmp1/tmp2
        else:
            acc[ii] = -1

    return th, acc

def conf_correct(p, mask, th=0.5):
    mask = (mask > 0)
    pred_mask = (p >= th)
    consider = (pred_mask == mask)
    conf = np.maximum(p, 1.0 - p)
    conf_corr = np.ma.array(conf, mask=(1-consider))
    conf_uncorr = np.ma.array(conf, mask=consider)
    conf_corr_avg = np.ma.average(conf_corr)
    conf_uncorr_avg = np.ma.average(conf_uncorr)

    return conf_corr_avg, conf_uncorr_avg


