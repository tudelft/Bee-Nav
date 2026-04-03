import os
import csv
from PIL import Image
import numpy as np
import re
import json

class WrapAndRotateTransform:
    def __init__(self, vector_map):
        self.vector_map = vector_map

    def __call__(self, image, filename):
        vector = self.vector_map[filename]
        image_array = np.array(image)
        return self.create_wrapped_images(image_array, vector)

    def create_wrapped_images(self, image_array, vector):
        # height, width = image_array.shape
        height, width, channels = image_array.shape
        pixels_per_degree = width / 360
        wrapped_images = []

        for degree in range(1,360,5):
            shift_amount = int(degree * pixels_per_degree)
            vector_new = self.rotate_vector(vector, -degree)
            left_part = image_array[:, :shift_amount]
            right_part = image_array[:, shift_amount:]
            wrapped_image = np.concatenate((right_part, left_part), axis=1)
            wrapped_image = Image.fromarray(wrapped_image)
            wrapped_images.append((wrapped_image, vector_new))
            #wrapped_images.append((vector_new))
        return wrapped_images

    def rotate_vector(self, vector, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians),  np.cos(angle_radians)]
        ])
        rotated_vector = np.dot(rotation_matrix, vector[:2])
        rotated_vector = np.concatenate((rotated_vector, vector[2:]))
        return rotated_vector

def get_vector_from_string(vector_str):
    # print(vector_str)
    if vector_str.startswith('['):
        vector_str = vector_str.strip('[]')
        # # replace multiple whitespaces with single whitespace:
        # vector_str = re.sub(' +', ' ', vector_str)
        elements = vector_str.split(' ')
        # remove elements that are empty strings:
        elements = [element for element in elements if element]
    else:
        vector_str = vector_str.strip('()')
        vector_str = vector_str.replace(' ', '')
        elements = vector_str.split(',')
    return np.array([float(num) for num in elements])

def load_dataset_from_csv(csv_file_path):

    # lists with filenames and targets: 
    filenames = []
    target_x = []
    target_y = []
    # maps as made in the original code, i.e., dictionaries indexed by filename:
    position_map  = {}
    rotation_map = {}
    target_map = {}
    noisy_position_map = {}
    noisy_rotation_map = {}
    noisy_target_map = {}
    noise = False
    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row
        for row in reader:
            if len(row) == 4:
                filename, position, rotation, target = row
                # maps:
                position = get_vector_from_string(position)
                position_map[filename] = position
                rotation = get_vector_from_string(rotation)
                rotation_map[filename] = rotation
                target = get_vector_from_string(target)
                target_map[filename] = target
                # data set to be used:
                target_x.append(target[0])
                target_y.append(target[1])
                filenames.append(filename)
                
            elif len(row) == 7:
                if noise == False:
                    noise = True
                filename, position, rotation, target, noisy_position, noisy_rotation, noisy_target = row
                # maps: 
                position_map[filename] = get_vector_from_string(position)
                rotation_map[filename] = get_vector_from_string(rotation)
                target = get_vector_from_string(target)
                target_map[filename] = target
                noisy_position_map[filename] = get_vector_from_string(noisy_position)
                noisy_rotation_map[filename] = get_vector_from_string(noisy_rotation)
                noisy_target_map[filename] = get_vector_from_string(noisy_target)
                # data set:
                # target is used, as the robot thinks it is at the position, and not the noisy position:
                target_x.append(target[0])
                target_y.append(target[1])
                filenames.append(filename)
    if noise:
        return filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map
    else:
        return filenames, target_x, target_y, position_map, rotation_map, target_map, None, None, None

def crop_images(output_folder, crop_box_params=(0, 312, 1024, 712)):
    # Crop images in the output folder
    for filename in os.listdir(output_folder):
        if filename.endswith('.png'):
            image_path = os.path.join(output_folder, filename)
            image = Image.open(image_path).convert('RGB')
            crop_box = crop_box_params
            image = image.crop(crop_box)
            image.save(image_path)

