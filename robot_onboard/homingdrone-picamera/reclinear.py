import cv2
import numpy as np
import pickle
import os

def read_image(file_name):
    image_org = cv2.imread(file_name, cv2.IMREAD_COLOR)
    return image_org

def apply_mask(image_gray, mask):
    image_masked = cv2.bitwise_and(image_gray, image_gray, mask=mask)
    return image_masked

def convert_image(image_masked, x, y, r):
    image_polar = cv2.linearPolar(image_masked, (x, y), r, cv2.WARP_FILL_OUTLIERS)
    return image_polar

def after_process(image_polar):
    image_rectilinear = cv2.rotate(image_polar, cv2.ROTATE_90_COUNTERCLOCKWISE)
    image_flipped = np.flipud(image_rectilinear)
    image_reshaped = cv2.resize(image_flipped[840:,:], (1800, 192))
    image_final = np.fliplr(image_reshaped)
    image_final = cv2.resize(image_final[15:165, :], (1800,192))
    return image_final

def rotate_image(image_final):
    image_rotate = np.concatenate((image_final[:,450:], image_final[:,:450]), axis=1)
    return image_rotate

def get_mask():
    with open("./mask.pkl", "rb") as f:
        mask_file = pickle.load(f)
    mask = mask_file['mask']
    x,y,r = mask_file['x'], mask_file['y'], mask_file['r']
    return mask, x, y, r
