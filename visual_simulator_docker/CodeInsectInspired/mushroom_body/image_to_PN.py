import numpy as np
import cv2
import os
from skimage import filters
from PIL import Image

# --- Configuration ---
# NOTE: The original code uses a variable called self._params_vision.
# We must define the values it holds for this script to work.
# Assuming _params_vision is [resolution, sigma]
# Common values for vision tasks like this are:
VISION_RESOLUTION = 32  # The side length (in pixels) for the resampled image
GAUSSIAN_SIGMA = 1.0    # The sigma value for the Gaussian blur

# Folder paths — update these to point to your local data directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
INPUT_FOLDER = os.path.join(PROJECT_ROOT, 'forest', 'forest_6_trees_home_circle_1000_mushroom', 'Replicator', 'rgb')
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, 'forest', 'forest_6_trees_home_circle_1000_mushroom', 'Replicator', 'PN')
# ---------------------

# create the output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)


def create_pn_vector(image_path: str, resolution: int, sigma: float) -> np.ndarray:
    """
    Converts a single image file into a Projection Neuron (PN) vector 
    using the provided image processing pipeline.

    Args:
        image_path: Path to the input image file.
        resolution: The target square side length for resizing.
        sigma: The sigma for the Gaussian filter.

    Returns:
        A 1D numpy array representing the PN vector.
    """
    try:
        # 1. Load the image using PIL (handles various formats better than cv2 for initial load)
        image_pil = Image.open(image_path).convert('RGB')
        image = np.array(image_pil)
    except Exception as e:
        print(f"Error loading {image_path}: {e}")
        return None

    print(f"[load] shape={image.shape}, dtype={image.dtype}")

    # --- Start of the original PN creation logic ---

    # image = image[:, :, 1] - Takes the Green channel (index 1)
    if image.ndim == 3:
        image = image[:, :, 1]
        print(f"[channel] took green channel -> shape={image.shape}, dtype={image.dtype}")
    else:
        print(f"[channel] image already single-channel -> shape={image.shape}, dtype={image.dtype}")
    
    # image = image.astype(np.uint8) - Ensure correct type (mostly for consistency)
    image = image.astype(np.uint8)
    print(f"[astype] converted to uint8 -> shape={image.shape}, dtype={image.dtype}")
    
    # Gaussian filter
    # Note: skimage filters expect float image data, which Gaussian outputs
    image = filters.gaussian(image, sigma=sigma)
    print(f"[gaussian] after gaussian -> shape={image.shape}, dtype={image.dtype}, min={image.min():.6f}, max={image.max():.6f}")
    
    # Resizing
    img_resampled = cv2.resize(
        image, (int(resolution), int(resolution)), interpolation=cv2.INTER_NEAREST
    )
    print(f"[resize] resampled -> shape={img_resampled.shape}, dtype={img_resampled.dtype}")
    
    # Sobel filter for edge detection
    img_sobel = filters.sobel(img_resampled)
    print(f"[sobel] after sobel -> shape={img_sobel.shape}, dtype={img_sobel.dtype}, min={img_sobel.min():.6f}, max={img_sobel.max():.6f}")
    
    ra = resolution / 2
    cx = resolution / 2
    cy = resolution / 2
    
    # create a meshgrid of indices for the image
    x, y = np.meshgrid(np.arange(img_sobel.shape[0]), np.arange(img_sobel.shape[1]))
    print(f"[meshgrid] x.shape={x.shape}, y.shape={y.shape}")
    
    # calculate distances from center of image
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    print(f"[dist] dist.shape={dist.shape}, dist.min={dist.min():.6f}, dist.max={dist.max():.6f}")
    
    # create boolean masks for radii (outer_mask = circular crop)
    outer_mask = dist <= ra
    print(f"[mask] outer_mask.shape={outer_mask.shape}, True_count={outer_mask.sum()}")
    
    # apply masks to image, flattening to the 1D PN vector
    pn_vector_raw = img_sobel[outer_mask]
    print(f"[mask_apply] pn_vector_raw.shape={pn_vector_raw.shape}, dtype={pn_vector_raw.dtype}, min={pn_vector_raw.min():.6f}, max={pn_vector_raw.max():.6f}")
    
    # Normalization (Min-Max scaling to 0-255, then casting to uint8)
    # This is applied to the final PN vector (pn)
    pn_min = pn_vector_raw.min()
    pn_max = pn_vector_raw.max()
    
    if pn_max - pn_min > 0:
        # Scale to 0-255 range
        pn = ((pn_vector_raw - pn_min) * (255.0 / (pn_max - pn_min))).astype('uint8')
        print(f"[normalize] scaled to 0-255 -> shape={pn.shape}, dtype={pn.dtype}, min={pn.min()}, max={pn.max()}")
    else:
        # Handle case where all values are the same (flat image)
        pn = np.zeros_like(pn_vector_raw, dtype='uint8')
        print(f"[normalize] flat image -> created zeros -> shape={pn.shape}, dtype={pn.dtype}")
        
    # --- End of the original PN creation logic ---
    
    return pn.flatten() # Return the 1D PN vector


def process_image_folder(input_dir: str, output_dir: str, resolution: int, sigma: float):
    """
    Processes all image files in a directory and saves the resulting PN vectors.
    """
    # print(f"--- Starting PN Vector Generation ---")
    # print(f"Input: {input_dir}, Output: {output_dir}")
    # print(f"Params: Resolution={resolution}, Sigma={sigma}\n")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_count = 0
    
    # List all files in the input folder
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_path = os.path.join(input_dir, filename)
            
            # 1. Create the PN vector
            pn_vector = create_pn_vector(image_path, resolution, sigma)
            
            if pn_vector is not None:
                # 2. Define the output file name (e.g., 'image1.png' -> 'image1.npy')
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f'{base_name}.npy')
                
                # 3. Save the numpy array
                np.save(output_path, pn_vector)
                
                print(f"Processed: {filename} -> Saved to {output_path} (Size: {pn_vector.shape[0]})")
                processed_count += 1

    print(f"\n--- Processing Complete! {processed_count} images processed. ---")
    
    if processed_count == 0:
         print(f"Check that your '{input_dir}' folder contains supported image files.")


if __name__ == '__main__':
    # Execute the folder processing script
    process_image_folder(
        input_dir=INPUT_FOLDER,
        output_dir=OUTPUT_FOLDER,
        resolution=VISION_RESOLUTION,
        sigma=GAUSSIAN_SIGMA
    )