from scipy import ndimage, misc
import numpy as np
import os
import cv2.cv2 as cv2

path = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/data/bla'
outPath = 'C:/Users/ralf-/OneDrive/!Uni/Seminar/data/blubb'


def rot_img(img):
    height = img.shape[0]
    width = img.shape[1]
    rotated_img = np.ndarray([width, height, img.shape[2]], dtype="uint8")
    # for w in range(width):
    #     for h in range(height):
    #         rotated_img[w, h, :] = img[height - h - 1, w, :]
    for h in range(height):
        rotated_img[:, height-h-1, :] = img[h, :, :]
    return rotated_img

def main():
    # iterate through the names of contents of the folder
    for index, file in enumerate(os.listdir(path)):
        # create the full input path and read the file
        input_path = path + '/' + str(index+83) + '.jpg'
        image_to_rotate = cv2.imread(input_path)

        # rotate the image
        if image_to_rotate.shape[0] > image_to_rotate.shape[1]:
            rotated = rot_img(image_to_rotate)
        else:
            rotated = image_to_rotate

        # create full output path, 'example.jpg'
        # becomes 'rotate_example.jpg', save the file to disk
        fullpath = os.path.join(outPath, ''.join([str(index+83), '.jpg']))
        cv2.imwrite(fullpath, rotated)


if __name__ == '__main__':
    main()

