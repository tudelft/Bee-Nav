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
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        image = np.flipud(image)
        image = cv2.resize(image[840:], (1800, 192))
        image = np.fliplr(image)
        return cv2.resize(image[15:165, :], (1800, 192))

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
        os.system(f"cp {self.csv_location} {self.output_location}")

        for file in os.listdir(self.file_location):
            if not file.endswith('.jpg'):
                continue
            self.mask, self.x, self.y, self.r = self._load_mask()

            file_id = file.split('.')[0]

            file_name = os.path.join(self.file_location, file)

            image = self._read_image(file_name)
            if self.wind_correction:
                pitch = self.df[self.df['path_idx'] == file.split('.')[0]]['pitch'].values[0]
                roll = self.df[self.df['path_idx'] == file.split('.')[0]]['roll'].values[0]
                self.x, self.y = self._wind_correct(image, pitch, roll)

            self._process_image(image, file_id)


if __name__ == "__main__":
    file_location = ""
    csv_location = ''
    output_location = f""

    processor = RectilinearProcessor(file_location, csv_location, output_location, wind_correction=True, crop_type='upper', input_size='192x1800')
    processor.run()
