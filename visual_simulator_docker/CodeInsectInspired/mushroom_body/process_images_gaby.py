import numpy as np
import cv2
from skimage import filters
from matplotlib import pyplot as plt

def create_pn(image, resolution=32, sigma=3, show_images=False):
    """
    Function converting the raw image from the sensors
    to the corresponding Neural Projection.
    Adapted from Gaby's antcar_sim.py.
    """
    # Check if image is BGR or RGB. Assuming BGR (cv2 default) or RGB.
    # Gaby's code: image[:, :, 1] -> Green channel.
    # If input is (H, W, 3), this works.
    
    image_green = image[:, :, 1]
    image_green = image_green.astype(np.uint8)

    # Gaussian Blur
    image_blurred = filters.gaussian(image_green, sigma=sigma)
    
    # Resize
    img_resampled = cv2.resize(
        image_blurred, (int(resolution), int(resolution)), interpolation=cv2.INTER_NEAREST
    )
    
    # Sobel filter for edge detection
    img_sobel = filters.sobel(img_resampled)
    
    if show_images:
        # show as heat plots in a single window with subplots:
        plt.figure(figsize=(10, 8))
        plt.subplot(1, 4, 1)
        plt.title("Original Green Channel")
        plt.imshow(image_green, cmap='gray')
        plt.subplot(1, 4, 2)
        plt.title("Blurred Image")
        plt.imshow(image_blurred, cmap='gray')
        plt.subplot(1, 4, 3)
        plt.title("Resampled Image")
        plt.imshow(img_resampled, cmap='gray')
        plt.subplot(1, 4, 4)
        plt.title("Sobel Image")
        plt.imshow(img_sobel, cmap='gray')
        plt.show()

    # Circular Mask
    ra = resolution / 2
    cx = resolution / 2
    cy = resolution / 2
    
    # create a meshgrid of indices for the image
    x, y = np.meshgrid(np.arange(img_sobel.shape[0]), np.arange(img_sobel.shape[1]))
    
    # calculate distances from center of image
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # create boolean masks for radii
    outer_mask = dist <= ra
    
    # apply masks to image
    pn = img_sobel[outer_mask]
    
    if show_images:
        masked_image = np.zeros_like(img_sobel)
        masked_image[outer_mask] = img_sobel[outer_mask]
        plt.figure()
        plt.title("Masked Image")
        plt.imshow(masked_image, cmap='gray')
        plt.show()

    # Normalize
    if img_sobel.max() - img_sobel.min() > 0:
        img_sobel = ((img_sobel - img_sobel.min()) * (1/(img_sobel.max() - img_sobel.min()) * 255)).astype('uint8')
    else:
        img_sobel = np.zeros_like(img_sobel, dtype='uint8')
        
    if pn.max() - pn.min() > 0:
        pn = ((pn - pn.min()) * (1/(pn.max() - pn.min()) * 255)).astype('uint8')
    else:
        pn = np.zeros_like(pn, dtype='uint8')
    
    if show_images:
        plt.figure()
        plt.title("Normalized PN")
        plt.imshow(img_sobel, cmap='gray')
        plt.show()

    return pn, img_sobel

def rotate_image(image, angle):
    """
    Rotates an image by a given angle (degrees).
    Positive angle = Clockwise rotation.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
