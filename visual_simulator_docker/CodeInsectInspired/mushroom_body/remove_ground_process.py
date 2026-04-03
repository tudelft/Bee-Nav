import os
from PIL import Image

def process_images(input_folder, output_folder):
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # 2. Iterate through files in the input folder
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_extensions)]
    
    if not files:
        print("No images found in the input folder.")
        return

    print(f"Found {len(files)} images. Processing...")

    for filename in files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                original_width, original_height = img.size
                
                # --- CALCULATION ---
                # We want to remove the outer 1/3 of the radius.
                # This means we want to KEEP the inner 2/3 of the image.
                keep_ratio = 1/2
                
                crop_width = original_width * keep_ratio
                crop_height = original_height * keep_ratio
                
                # Calculate coordinates for the center crop
                left = (original_width - crop_width) / 2
                top = (original_height - crop_height) / 2
                right = (original_width + crop_width) / 2
                bottom = (original_height + crop_height) / 2
                
                # --- CROP ---
                cropped_img = img.crop((left, top, right, bottom))
                
                # --- RESIZE ---
                # Resize back to original dimensions using high-quality resampling
                resized_img = cropped_img.resize((original_width, original_height), Image.Resampling.LANCZOS)
                
                # --- SAVE ---
                # Preserve quality for JPEGs (95 is usually sufficient)
                resized_img.save(output_path, quality=95)
                print(f"Processed: {filename}")

        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("Done!")

# --- CONFIGURATION ---
# Update these default paths or pass --input_dir / --output_dir on the command line
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
default_input_dir = os.path.join(PROJECT_ROOT, 'cars', 'brick_world_1000_mushroom_norotation_grid_sparse_3m', 'Replicator', 'rgb')
default_output_dir = os.path.join(PROJECT_ROOT, 'cars', 'brick_world_1000_mushroom_norotation_grid_sparse_3m', 'Replicator', 'rgb_no_ground')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default=default_input_dir)
    parser.add_argument("--output_dir", type=str, default=default_output_dir)
    args = parser.parse_args()
    process_images(args.input_dir, args.output_dir)