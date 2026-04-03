import os
import cv2
import numpy as np
import argparse
from skimage import filters
from PIL import Image
from tqdm import tqdm
import glob

# Configuration
VISION_RESOLUTION = 32
GAUSSIAN_SIGMA = 1.0

def convert_to_fisheye(img):
    """
    Converts a perspective image to a fisheye-like projection.
    Based on logic from fisheye_process.py
    """
    h, w = img.shape[:2]
    output_size = min(h, w)
    out_h = out_w = output_size
    
    dest_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    
    # Pre-compute coordinates for speed
    y_inds, x_inds = np.indices((out_h, out_w))
    dx = x_inds - out_w / 2
    dy = y_inds - out_h / 2
    r = np.sqrt(dx**2 + dy**2)
    max_r = out_w / 2
    
    # Mask for inside circle
    mask = r <= max_r
    
    theta = r[mask] / max_r * (np.pi / 2)
    phi = np.arctan2(dy[mask], dx[mask])
    
    src_x = (w * (phi + np.pi) / (2 * np.pi)).astype(int)
    src_y = (h * theta / (np.pi / 2)).astype(int)
    
    # Clip coordinates
    src_x = np.clip(src_x, 0, w - 1)
    src_y = np.clip(src_y, 0, h - 1)
    
    dest_img[y_inds[mask], x_inds[mask]] = img[src_y, src_x]
    
    return dest_img

def create_pn_vector(image, resolution=VISION_RESOLUTION, sigma=GAUSSIAN_SIGMA):
    """
    Converts an image (RGB numpy array) to a PN vector.
    Based on logic from image_to_PN.py
    """
    # Take Green channel
    if image.ndim == 3:
        image = image[:, :, 1]
    
    image = image.astype(np.float32) # filters.gaussian expects float
    
    # Gaussian filter
    image = filters.gaussian(image, sigma=sigma)
    
    # Resizing
    img_resampled = cv2.resize(
        image, (int(resolution), int(resolution)), interpolation=cv2.INTER_NEAREST
    )
    
    # Sobel filter
    img_sobel = filters.sobel(img_resampled)
    
    ra = resolution / 2
    cx = resolution / 2
    cy = resolution / 2
    
    # Create meshgrid
    x, y = np.meshgrid(np.arange(img_sobel.shape[0]), np.arange(img_sobel.shape[1]))
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    
    # Mask
    outer_mask = dist <= ra
    pn_vector_raw = img_sobel[outer_mask]
    
    # Normalize
    pn_min = pn_vector_raw.min()
    pn_max = pn_vector_raw.max()
    
    if pn_max - pn_min > 0:
        pn = ((pn_vector_raw - pn_min) * (255.0 / (pn_max - pn_min))).astype('uint8')
    else:
        pn = np.zeros_like(pn_vector_raw, dtype='uint8')
        
    return pn.flatten()

def process_folder(input_dir, output_pn_dir, output_fisheye_dir=None):
    """
    Processes all images in input_dir.
    """
    if not os.path.exists(output_pn_dir):
        os.makedirs(output_pn_dir)
    
    if output_fisheye_dir and not os.path.exists(output_fisheye_dir):
        os.makedirs(output_fisheye_dir)
        
    extensions = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    print(f"Found {len(files)} images in {input_dir}")
    
    for path in tqdm(files, desc="Processing Images"):
        filename = os.path.basename(path)
        base_name = os.path.splitext(filename)[0]
        
        # Check if output already exists
        pn_out_path = os.path.join(output_pn_dir, f'{base_name}.npy')
        if os.path.exists(pn_out_path):
            continue

        try:
            # Load Image
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Could not read {path}")
                continue
                
            # Convert to Fisheye
            fisheye_img = convert_to_fisheye(img)
            
            # Save Fisheye (Optional)
            if output_fisheye_dir:
                cv2.imwrite(os.path.join(output_fisheye_dir, filename), fisheye_img)
            
            # Convert to PN
            pn_vector = create_pn_vector(fisheye_img)
            
            # Save PN
            np.save(pn_out_path, pn_vector)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images for Mushroom Body simulation.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing RGB images")
    parser.add_argument("--output_pn_dir", type=str, required=True, help="Directory to save PN vectors")
    parser.add_argument("--output_fisheye_dir", type=str, default=None, help="Directory to save intermediate Fisheye images (optional)")
    
    args = parser.parse_args()
    
    process_folder(args.input_dir, args.output_pn_dir, args.output_fisheye_dir)
