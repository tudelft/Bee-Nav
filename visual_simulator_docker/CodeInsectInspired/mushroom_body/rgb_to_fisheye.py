import numpy as np


def rgb_to_fisheye(img):
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
            src_x = int(w * (phi + np.pi) / (2 * np.pi)) % w
            src_y = int(h * theta / (np.pi / 2))
            if 0 <= src_x < w and 0 <= src_y < h:
                dest_img[y, x] = img[src_y, src_x]

    return dest_img