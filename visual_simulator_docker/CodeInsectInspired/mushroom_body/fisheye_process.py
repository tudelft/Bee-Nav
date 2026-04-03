import os
import glob
import cv2
import numpy as np


def process_fisheye(input_dir, output_dir):
    # 1. Create the output folder if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    extensions = ('*.png', '*.jpg', '*.jpeg')
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files:
        print("No images found in", input_dir)
    else:
        for path in files:
            img = cv2.imread(path)
            if img is None:
                print("Skipping (can't read):", path)
                continue

            h, w = img.shape[:2]

            # Output size (square for circular fisheye)
            output_size = min(h, w)
            out_h = out_w = output_size

            # Prepare blank destination image
            dest_img = np.zeros((out_h, out_w, 3), dtype=np.uint8)

            # Map coordinates from the circle to spherical coordinates
            for y in range(out_h):
                for x in range(out_w):
                    dx = x - out_w / 2
                    dy = y - out_h / 2
                    r = np.sqrt(dx**2 + dy**2)
                    max_r = out_w / 2
                    if r > max_r:
                        continue  # only positions inside the circle
                    theta = r / max_r * (np.pi / 2)  # fisheye projection
                    phi = np.arctan2(dy, dx)
                    # Convert to equirectangular
                    src_x = int(w * (phi + np.pi) / (2 * np.pi))
                    src_y = int(h * theta / (np.pi / 2))
                    if 0 <= src_x < w and 0 <= src_y < h:
                        dest_img[y, x] = img[src_y, src_x]

            # Save to output directory
            filename = os.path.basename(path)
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, dest_img)
            print("Processed and saved:", output_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # Update these default paths or pass --input_dir / --output_dir on the command line
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    default_input = os.path.join(project_root, 'cars', 'brick_world_1000_mushroom_norotation_grid_sparse_3m', 'Replicator', 'rgb')
    default_output = os.path.join(project_root, 'cars', 'brick_world_1000_mushroom_norotation_grid_sparse_3m', 'Replicator', 'rgb_fisheye')
    
    parser.add_argument("--input_dir", type=str, default=default_input)
    parser.add_argument("--output_dir", type=str, default=default_output)
    args = parser.parse_args()
    
    process_fisheye(args.input_dir, args.output_dir)
