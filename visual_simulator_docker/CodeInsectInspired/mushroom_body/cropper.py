import cv2
import numpy as np

def crop_and_zoom_process(img, keep_ratio=1/2):
    """
    Crops the outer 1/3 of the image (keeping the inner 2/3)
    and resizes it back to the original resolution.
    
    Args:
        img: A numpy array representing the image (H, W, C)
    Returns:
        dest_img: The processed numpy array
    """
    # 1. Get dimensions
    h, w = img.shape[:2]
    
    # 2. Calculate the crop size
    # Removing outer 1/3 radius = keeping inner 2/3 diameter
    crop_h = int(h * keep_ratio)
    crop_w = int(w * keep_ratio)
    
    # 3. Calculate center coordinates to slice
    # We want to start at (Total - Crop) / 2
    y_start = (h - crop_h) // 2
    x_start = (w - crop_w) // 2
    
    y_end = y_start + crop_h
    x_end = x_start + crop_w
    
    # 4. Perform the Crop (Slicing)
    cropped_section = img[y_start:y_end, x_start:x_end]
    
    # 5. Resize back to original size
    # INTER_LANCZOS4 is usually best for upscaling/zooming to keep details sharp
    dest_img = cv2.resize(cropped_section, (w, h), interpolation=cv2.INTER_LANCZOS4)
    
    return dest_img