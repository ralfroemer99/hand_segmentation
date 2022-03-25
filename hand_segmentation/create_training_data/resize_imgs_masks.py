import cv2.cv2 as cv2
import os

img = 1

if img:
    # path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/02_rotated_images'
    # outPath = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/03_resized_images'
    path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/00_backgrounds/bla'
    outPath = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/00_backgrounds/bla'
    filetype = '.jpg'
else:
    path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/12_masks'
    outPath = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/13_resized_masks'
    filetype = '.png'

for index, file in enumerate(os.listdir(path)):
    # create the full input path and read the file
    input_path = path + '/' + str(index + 7) + filetype
    img = cv2.imread(input_path)

    resized_img = cv2.resize(img, (384, 288))

    fullpath = os.path.join(outPath, ''.join([str(index + 1), filetype]))
    cv2.imwrite(fullpath, resized_img)

