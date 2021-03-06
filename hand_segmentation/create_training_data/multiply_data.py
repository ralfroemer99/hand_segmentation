import cv2.cv2 as cv
import os
import numpy as np
from albumentations import GridDistortion
import skimage.exposure

img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/images/training_set'
img_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/images/dset11'
img_pathOut2 = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/images/dset12'
mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/masks/training_set'
mask_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/masks/dset11'
mask_pathOut2 = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/datasets/masks/dset12'
bg_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/00_backgrounds'
# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/bin/01_test_img'
# img_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/bin/02_test_mult_img'
# mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/bin/11_test_mask'
# mask_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/own_images/bin/12_test_mult_mask'
# img_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/bla'
# img_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper'
# mask_path = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper/blubb'
# mask_pathOut = 'C:/Users/ralf-/Documents/Python/SemanticSegmentation/paper'

k = 11    # How many augmented images are generated by one original images

# Get number of test_images
N = len(os.listdir(img_path))
if len(os.listdir(mask_path)) is not N:
    exit('No equal number of test_images and masks!')


def add_noise(img, noise_mask=None, mag=10):
    img = np.array(img, np.uint8)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    # noise_mask = np.random.randint((-1)*mag, mag, img.shape)
    if noise_mask is None:
        noise_mask = np.random.normal(0, mag, img.shape)
    img_hsv = img_hsv + noise_mask
    img_hsv[img_hsv > 255] = 255
    img_hsv[img_hsv < 0] = 0
    img_hsv = np.array(img_hsv, np.uint8)
    img = cv.cvtColor(img_hsv, cv.COLOR_HSV2BGR)
    return img, noise_mask


def change_background(img, mask, background):
    background[mask == 255] = img[mask == 255]
    # background[mask == 1] = background[mask == 1] + img[mask == 1]
    return background


def change_skin_color(img, mask, des_color):
    n_skins = 20        # Must be even! Always start from darkest skin color

    skin_colors = np.zeros((n_skins, 3), dtype=int)
    for ii in range(20):
        r = np.minimum(255, 15 + ii*15)
        g = np.round(11 + 11.5*ii).astype(int)
        b = 10 + 10*ii
        skin_colors[ii, :] = np.array([b, g, r])

    # Create random skin color. Darker skin colors have twice the probability of being generated as lighter skin colors
    which_skin_type = np.random.randint(0, 3)
    if which_skin_type == 0:
        which_skin = np.random.randint(n_skins/2 - 1, n_skins)
    else:
        which_skin = np.random.randint(0, n_skins/2 - 1)
    # which_skin = 0
    des_color = skin_colors[which_skin, :]
    img = np.array(img, np.uint8)
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    res_partial = np.copy(img)
    res_partial[mask[:, :, 0] > 0] = des_color
    alpha = np.interp(which_skin, [0, n_skins], [0.9, 0.1])
    res_final = cv.addWeighted(res_partial, alpha, img, 1-alpha, 0.0)
    return res_final


def random_erase(img):
    height = 288
    width = 384
    rel_limit_l = 0.1
    rel_limit_u = 0.3
    rectangle_height = np.random.randint(rel_limit_l*height, rel_limit_u*height)
    rectangle_width = np.random.randint(rel_limit_l*width, rel_limit_u*width)
    height_start = np.random.randint(0.1*height, 0.9*height-rectangle_height)
    width_start = np.random.randint(0.1, 0.9*width-rectangle_width)
    rectangle = np.random.normal(128, 50, (rectangle_height, rectangle_width, 3))
    img = np.array(img, np.uint8)
    img[height_start:height_start+rectangle_height, width_start:width_start+rectangle_width, :] = rectangle
    return img



for index in range(N):
    # create the full image path and read the image file
    full_path = img_path + '/' + str(index + 1) + '.jpg'
    img = cv.imread(full_path)

    # create the full mask path and read the mask file
    full_path = mask_path + '/' + str(index + 1) + '.png'
    mask = cv.imread(full_path)

    # Store changed images in 4D array
    imgs = np.empty((k, img.shape[0], img.shape[1], img.shape[2]))
    imgs[0] = img
    imgs2 = np.empty((k, img.shape[0], img.shape[1], img.shape[2]))
    imgs2[0] = img

    # Store changed masks in 4D array
    masks = np.empty((k, mask.shape[0], mask.shape[1], mask.shape[2]))
    masks[0] = mask
    masks2 = np.empty((k, mask.shape[0], mask.shape[1], mask.shape[2]))
    masks2[0] = mask

    # Double dataset
    # imgs[1] = imgs[0]
    # masks[1] = masks[0]

    # Only flip (k = 2)
    # tmp = np.random.randint(0, 2)
    # if tmp == 0:
    # imgs[1] = np.fliplr(img)
    # masks[1] = np.fliplr(mask)
    # else:
    # imgs[1] = np.flipud(img)
    # masks[1] = np.flipud(mask)

    # Only rotate (k = 2)
    # imgs[1] = np.fliplr(np.flipud(img))
    # masks[1] = np.fliplr(np.flipud(mask))

    # Only grid distortion (k = 2)
    # aug = GridDistortion(interpolation=cv.INTER_NEAREST, border_mode=cv.BORDER_CONSTANT, p=1.0)
    # distorted = aug(image=imgs[0], mask=masks[0])
    # imgs[1] = distorted['image']
    # masks[1] = distorted['mask']

    # Only noise (k=2)
    # imgs[1] = add_noise(imgs[0])
    # masks[1] = masks[0]               # Masks are left unchanged

    # Only new background (k=2)
    # tmp = np.random.randint(0, 10)
    # full_path = bg_path + '/' + str(tmp + 1) + '.jpg'
    # bg = cv.imread(full_path)
    # imgs[1] = change_background(img, mask, bg)
    # masks[1] = mask

    # Only color modification (k=2)
    # imgs[1] = change_skin_color(imgs[0], masks[0], (180, 128, 200))
    # masks[1] = masks[0]

    # Only random erase
    # imgs[1] = random_erase(imgs[0])
    # masks[1] = masks[0]

    # # FLIP: 2, 3, 4
    # imgs[1] = np.fliplr(img)
    # imgs[2] = np.flipud(img)
    # imgs[3] = np.fliplr(np.flipud(img))
    # masks[1] = np.fliplr(mask)
    # masks[2] = np.flipud(mask)
    # masks[3] = np.fliplr(np.flipud(mask))
    #
    # # ADD NOISE to images 2, 3, 4
    # for ii in range(4):
    #     imgs[ii + 1] = add_noise(imgs[ii + 1])
    #     masks[ii + 1] = masks[ii + 1]               # Masks are left unchanged
    #
    # # DISTORT images 2, 3, 4
    # aug = GridDistortion(interpolation=cv.INTER_NEAREST, border_mode=cv.BORDER_CONSTANT, p=1.0)
    # for ii in range(4):
    #     distorted = aug(image=imgs[ii + 1], mask=masks[ii + 1])
    #     imgs[ii + 1] = distorted['image']
    #     masks[ii + 1] = distorted['mask']
    #
    # # CHANGE BACKGROUND, flip/rotate randomly, add noise, distort grid --> 5, 6, 7, 8, 9, 10
    # for ii in range(10):
    #     full_path = bg_path + '/' + str(ii + 1) + '.jpg'
    #     bg = cv.imread(full_path)
    #     imgs[ii+4] = change_background(img, mask, bg)
    #     masks[ii+4] = mask
    #     tmp = np.random.randint(0, 3)
    #     if tmp == 0:
    #         imgs[ii + 4] = np.fliplr(imgs[ii + 4])
    #         masks[ii + 4] = np.fliplr(masks[ii + 4])
    #     elif tmp == 1:
    #         imgs[ii + 4] = np.flipud(imgs[ii + 4])
    #         masks[ii + 4] = np.flipud(masks[ii + 4])
    #     else:
    #         imgs[ii + 4] = np.fliplr(np.flipud(imgs[ii + 4]))
    #         masks[ii + 4] = np.fliplr(np.flipud(masks[ii + 4]))
    #     # Change skin color
    #     imgs[ii + 4] = change_skin_color(imgs[ii + 4], masks[ii + 4], (180, 128, 200))
    #
    #     # Add noise
    #     imgs[ii + 4] = add_noise(imgs[ii + 4], mag=10)
    #
    #     # Distort grid
    #     distorted = aug(image=imgs[ii + 4], mask=masks[ii + 4])
    #     imgs[ii + 4] = distorted['image']
    #     masks[ii + 4] = distorted['mask']

    # CHANGE BACKGROUND using all 10 additional backgrounds
    aug = GridDistortion(interpolation=cv.INTER_NEAREST, border_mode=cv.BORDER_CONSTANT, p=1.0)
    for ii in range(1, 11):
        full_path = bg_path + '/' + str(ii) + '.jpg'
        bg = cv.imread(full_path)
        img_tmp = change_background(img, mask, bg)
        mask_tmp = mask
        tmp = np.random.randint(0, 3)
        if tmp == 0:
            img_tmp = np.fliplr(img_tmp)
            mask_tmp = np.fliplr(mask_tmp)
        elif tmp == 1:
            img_tmp = np.flipud(img_tmp)
            mask_tmp = np.flipud(mask_tmp)
        else:
            img_tmp = np.fliplr(np.flipud(img_tmp))
            mask_tmp = np.fliplr(np.flipud(mask_tmp))
        # Color modification
        tmp = np.random.randint(0, 4)
        if tmp == 0:
            img_tmp = change_skin_color(img_tmp, mask_tmp, (180, 128, 200))
        # Grid distortion
        tmp = np.random.randint(0, 4)
        img_tmp2 = img_tmp
        mask_tmp2 = mask_tmp
        if tmp == 0:
            distorted = aug(image=img_tmp, mask=mask_tmp)
            img_tmp2 = distorted['image']
            mask_tmp2 = distorted['mask']
        # Noise
        tmp = np.random.randint(0, 4)
        if tmp == 0:
            img_tmp, noise_mask = add_noise(img_tmp)
            img_tmp2, _ = add_noise(img_tmp2, noise_mask)
        # Random erase
        tmp = np.random.randint(0, 4)
        if tmp == 0:
            img_tmp2 = random_erase(img_tmp2)
        imgs[ii] = img_tmp
        masks[ii] = mask_tmp
        imgs2[ii] = img_tmp2
        masks2[ii] = mask_tmp2

    # Save images and masks
    for ii in range(k):
        full_path = os.path.join(img_pathOut, ''.join([str(k * index + ii + 1), '.jpg']))
        if os.path.exists(full_path):
            os.remove(full_path)
        cv.imwrite(full_path, imgs[ii])
        full_path = os.path.join(mask_pathOut, ''.join([str(k * index + ii + 1), '.png']))
        if os.path.exists(full_path):
            os.remove(full_path)
        cv.imwrite(full_path, masks[ii])
        full_path = os.path.join(img_pathOut2, ''.join([str(k * index + ii + 1), '.jpg']))
        if os.path.exists(full_path):
            os.remove(full_path)
        cv.imwrite(full_path, imgs2[ii])
        full_path = os.path.join(mask_pathOut2, ''.join([str(k * index + ii + 1), '.png']))
        if os.path.exists(full_path):
            os.remove(full_path)
        cv.imwrite(full_path, masks2[ii])

