from oct2py import octave
import numpy as np
import cv2

datasetPath = 'C:/Users/ralf-/Documents/Python/Datasets/egohands_data'
octave.addpath(datasetPath)
octave.warning('off', 'all');


# i can be anything from 0 to 100 * 48 - 1 = 4799
def getSegMask(i):
    mask = np.zeros((720, 1280), dtype='uint8')

    hand1 = octave.getMask(i, 'my_left')
    hand2 = octave.getMask(i, 'my_right')
    hand3 = octave.getMask(i, 'your_left')
    hand4 = octave.getMask(i, 'your_right')
    hands = [hand1, hand2, hand3, hand4]

    for hand in hands:
        if hand is not None:
            hand = np.array(hand, dtype=int)
            cv2.fillPoly(mask, [hand], (255, 255, 255), 8)

    return mask