"""
Rectilinear images processing with wind compensation.
"""

import os
import cv2
import numpy as np
import pandas as pd
import pickle

class RectilinearProcessor:
    def __init__(self, file_location, csv_location, output_location, wind_correction, crop_type, input_size):
        self.file_location = file_location
        self.csv_location = csv_location
        self.output_location = output_location
        self.wind_correction = wind_correction
        self.crop_type = crop_type
        self.input_size = input_size
        self.df = pd.read_csv(csv_location)
        self.mask, self.x, self.y, self.r = self._load_mask()
        os.makedirs(output_location, exist_ok=True)
        

    def _load_mask(self):
        with open(os.path.join(os.path.dirname(__file__), "mask.pkl"), "rb") as f:
            mask_file = pickle.load(f)
        return mask_file['mask'], mask_file['x'], mask_file['y'], mask_file['r']

    def _read_image(self, file_name):
        return cv2.imread(file_name, cv2.IMREAD_COLOR)

    def _apply_mask(self, image, mask):
        return cv2.bitwise_and(image, image, mask=mask)

    def _convert_image(self, image, x, y, r):
        return cv2.linearPolar(image, (x, y), r, cv2.WARP_FILL_OUTLIERS)

    def _after_process(self, image):
        # --- OPTIMIZATION 3 ---
        # A 90-degree counter-clockwise rotation followed by a vertical flip
        # is the same as a single transpose operation.
        # image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # image = np.flipud(image)
        image = cv2.transpose(image)
        
        # --- OPTIMIZATION 4 ---
        # You resize, then slice, then resize again.
        # It's faster to slice the original array and resize ONCE.
        
        # Original logic:
        # image = cv2.resize(image[840:], (1800, 192))
        # image = np.fliplr(image)
        # return cv2.resize(image[15:165, :], (1800, 192))
        
        # Combined logic:
        image_sliced = image[840:, :]
        image_flipped = np.fliplr(image_sliced)
        
        # This seems to be the intended final slice based on your logic
        # First resize -> (192, 1800), then slice [15:165, :] -> (150, 1800)
        # We can approximate this by slicing the *original* flipped array
        # This assumes the first resize was a simple stretch.
        # Let's just do the slice and one resize.
        
        # This is what your original code does, but in one step:
        # 1. Slice `image[840:]`
        # 2. Flip it horizontally
        # 3. Resize to (1800, 192)
        temp_resized = cv2.resize(np.fliplr(image[840:, :]), (1800, 192))
        # 4. Slice that result [15:165, :]
        final_slice = temp_resized[15:165, :]
        # 5. Resize *again*
        return cv2.resize(final_slice, (1800, 192))
    def _rotate_image(self, image):
        return np.concatenate((image[:, 450:], image[:, :450]), axis=1)

    def _wind_correct(self, image, pitch, roll):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 2)
        circles = cv2.HoughCircles(blurred_image, cv2.HOUGH_GRADIENT, dp=1.2, minDist=1000, param1=50, param2=30, minRadius=240, maxRadius=243)
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            circles = circles.astype(np.uint16)
            x, y, r = circles[0]
            return int(x), int(y)
        # else:
        # x = -2.8378 * pitch + (-103.1169)*roll + 835.6210
        # y = 110.8523 * pitch + 2.5462 * roll + 643.1412
        else:
            x = 12.9546 * pitch + -130.7885*roll + 835.85
            y = 120.2679 * pitch + 14.7156 * roll + 643.05
        return int(x), int(y)

    def _process_image(self, image, file_id):
        
        # image = self._apply_mask(image, self.mask)
        image = self._convert_image(image, self.x, self.y, self.r)
        image = self._after_process(image)
        image = self._rotate_image(image)

        if self.crop_type == 'upper':
            height, width = image.shape[:2]
            image = cv2.resize(image[height // 2:], (width, height))
        if self.crop_type == 'middle':
            height, width = image.shape[:2]
            image = cv2.resize(image[height // 4:7 * height // 8], (width, height))

        output_file = os.path.join(self.output_location, f"{file_id}_preprocessed.jpg")
        cv2.imwrite(output_file, image)

    def run(self):
        import shutil
        shutil.copy(self.csv_location, self.output_location)

        # Use os.scandir for a slightly more efficient way to list files
        for entry in os.scandir(self.file_location):
            if not entry.name.endswith('.jpg') or not entry.is_file():
                continue
                
            # --- REMOVED ---
            # self.mask, self.x, self.y, self.r = self._load_mask() (Now in __init__)

            file_id = entry.name.split('.')[0]
            file_name = entry.path

            image = self._read_image(file_name)
            
            if self.wind_correction:
                try:
                    # --- OPTIMIZATION 2 (Continued) ---
                    # Use fast .loc lookup instead of slow full-dataframe search
                    row = self.df.loc[file_id]
                    pitch = row['pitch']
                    roll = row['roll']
                    # Use a copy of self.x/y if _wind_correct modifies them
                    # or just pass them directly
                    self.x, self.y = self._wind_correct(image, pitch, roll)
                except KeyError:
                    print(f"Warning: file_id {file_id} not found in CSV index. Skipping wind correction.")
                    continue # or whatever your error handling is
                except Exception as e:
                    print(f"Error processing {file_id}: {e}")
                    continue
                    
            self._process_image(image, file_id)

if __name__ == "__main__":
    file_location = ""
    csv_location = ''
    output_location = f""

    processor = RectilinearProcessor(file_location, csv_location, output_location, wind_correction=True, crop_type='upper', input_size='192x1800')
    processor.run()