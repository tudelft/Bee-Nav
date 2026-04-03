import os
import csv
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import torchvision.transforms.functional as TF
import pandas as pd
from insect_utils.augment_and_save import load_dataset_from_csv
from insect_utils.flight_path_functions import rotate_vector_by_yaw
import pickle
from matplotlib import pyplot as plt



def augment_image_vector(image_array, vector, degree):
    channels, height, width = image_array.shape
    pixels_per_degree = width / 360
    shift_amount = int(degree * pixels_per_degree)
    angle_radians = np.radians(degree) 
    rotation_matrix = np.array([
        [np.cos(angle_radians), -np.sin(angle_radians)],
        [np.sin(angle_radians),  np.cos(angle_radians)]
    ])
    rotated_vector = np.dot(rotation_matrix, vector)
    vector_new = torch.from_numpy(rotated_vector).float()
    left_part = image_array[:, :,  :shift_amount]
    right_part = image_array[:, :, shift_amount:]
    wrapped_image = torch.concatenate((right_part, left_part), axis=2)
    return wrapped_image, vector_new

class CustomDataset(Dataset):
    def __init__(self, image_folder, csv_file = None, image_names = None, target_x = None, target_y = None, transform=None, augmentation=False):
        
        if csv_file is not None:
            self.data = pd.read_csv(csv_file)
        elif image_names is not None:
            self.data = pd.DataFrame({
                'image name': image_names,
                'target_x': target_x,
                'target_y': target_y
            })
        else:
            raise ValueError("Either csv_file or image_names and labels must be provided")
        
        self.image_folder = image_folder
        self.transform = transform
        self.augmentation = augmentation

    def __len__(self):
        return len(self.data)

    def rotate_vector(self, vector, angle_degrees):
        angle_radians = np.radians(angle_degrees)
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians)],
            [np.sin(angle_radians),  np.cos(angle_radians)]
        ])
        rotated_vector = np.dot(rotation_matrix, vector[:2])
        rotated_vector = np.concatenate((rotated_vector, vector[2:]))
        return rotated_vector

    def rotate_image_and_vector(self, image_array, vector, degree):
        channels, height, width = image_array.shape
        pixels_per_degree = width / 360
        shift_amount = int(degree * pixels_per_degree)
        vector_new = self.rotate_vector(vector, degree) 
        vector_new = torch.from_numpy(vector_new).float()
        left_part = image_array[:, :,  :shift_amount]
        right_part = image_array[:, :, shift_amount:]
        wrapped_image = torch.concatenate((right_part, left_part), axis=2)
        
        return wrapped_image, vector_new

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_folder, self.data.iloc[idx, 0])
        image = Image.open(img_name).convert('RGB')
        
        label = self.data.iloc[idx, 1:].values.astype('float')
        label = torch.tensor(label, dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        if self.augmentation:
            # We will rotate the image, and the target vector:
            angle = np.random.randint(0, 360)
            new_image, new_label = self.rotate_image_and_vector(image, label, angle)
            image = new_image
            label = new_label
            

        return image, label

def make_loader(dataset, shuffle, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

def get_data_loader(data_folder, image_folder, csv_filename):
    filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map =  \
        load_dataset_from_csv(os.path.join(data_folder, csv_filename))
    
    full_dataset = CustomDataset(image_folder=image_folder, image_names = filenames, target_x=target_x, target_y = target_y,  transform=transforms.ToTensor(), \
                                 augmentation=False)

    loader =  make_loader(full_dataset, shuffle=False, batch_size=1)

    return loader, filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map

def check_ground_truth_rotation(position_map, rotation_map, target_map):
    # convert position map dict to list:
    position_map = [position_map[key] for key in sorted(position_map.keys())]
    rotation_map = [rotation_map[key] for key in sorted(rotation_map.keys())]
    target_map = [target_map[key] for key in sorted(target_map.keys())]

    for i in range(len(position_map)):
        pos = position_map[i]
        rot = rotation_map[i]
        target = target_map[i]
        rot_target = rotate_vector_by_yaw(target, rot[2])
        target_pos = pos[:2] + rot_target
        if abs(target_pos[0]) > 1E-4 or abs(target_pos[1]) > 1E-4:
            print("Error in ground truth rotation")
            print(f"Position: {pos}, Rotation: {rot}, Target: {target}, Target pos: {target_pos}")
            raise ValueError("Ground truth rotation check failed")
    
    print("Ground truth rotation check passed")

def show_image_and_target(image, target, prediction = None):
    if type(image) is torch.Tensor:
        image = TF.to_pil_image(image)
    width, height = image.size
    pixels_per_degree = width / 360

    angle = 360.0 - np.arctan2(target[1], target[0]) * 180 / np.pi
    if angle > 360:
        angle -= 360
    elif angle < 0:
        angle += 360
    pixels_angle_target = int(angle * pixels_per_degree)

    if prediction is not None:
        angle = 360.0 - np.arctan2(prediction[1], prediction[0]) * 180 / np.pi
        if angle > 360:
            angle -= 360
        elif angle < 0:
            angle += 360
        pixels_angle_prediction = int(angle * pixels_per_degree)

    plt.figure()
    plt.imshow(image)
    plt.plot([pixels_angle_target, pixels_angle_target], [0, height], 'g')
    if prediction is not None:
        plt.plot([pixels_angle_prediction, pixels_angle_prediction], [0, height], 'r')
    plt.show()

def inspect_dataset(config, n_examples = 10, load_indices = True, model = None):

    # Load the data set from the CSV file:
    filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map =  \
        load_dataset_from_csv(os.path.join(config["dataset_folder"], config['csv_filename']))
    
    # Create and split the processed dataset
    image_folder = config["image_folder"]
    full_dataset = CustomDataset(image_folder=image_folder, image_names = filenames, target_x=target_x, target_y = target_y,  transform=transforms.ToTensor() \
                                 , augmentation=False)
    
    indices_pkl_name = os.path.join(config["dataset_folder"], "indices.pkl")
    if load_indices and not os.path.exists(indices_pkl_name):
        load_indices = False

    if load_indices:
        # after training, we want to load the same indices:
        with open(indices_pkl_name, "rb") as file:
            indices = pickle.load(file)
        train_indices = indices["train_indices"]
        val_indices = indices["val_indices"]
        test_indices = indices["test_indices"]
    else:
        dataset_size = len(full_dataset)
        train_size = int(config["split_ratio"] * len(full_dataset))
        test_size = len(full_dataset) - train_size
        val_size = int((1-config["split_ratio"]) * train_size)

        # Generate shuffled indices manually
        indices = torch.randperm(dataset_size).tolist()
        test_indices = indices[train_size:]
        val_indices = indices[:val_size]
        train_indices = indices[val_size:train_size]

    if model is not None:
        model.eval()

    for ex in range(n_examples):
        ind = train_indices[ex]
        image, label = full_dataset[ind]

        degree = np.random.randint(0, 360)
        image, label = augment_image_vector(image, label, degree)

        if model is not None:
            t_image = image.unsqueeze(0)
            # Run the model:
            prediction = model(t_image)

        # show the image, and draw a line at the target location:
        image = TF.to_pil_image(image)
        width, height = image.size
        pixels_per_degree = width / 360

        angle = 360.0 - np.arctan2(label[1], label[0]) * 180 / np.pi
        if angle > 360:
            angle -= 360
        elif angle < 0:
            angle += 360
        pixels_angle_target = int(angle * pixels_per_degree)

        plt.figure()
        plt.imshow(image)
        plt.plot([pixels_angle_target, pixels_angle_target], [0, height], 'g')

        if model is not None:
            angle = 360.0 - np.arctan2(prediction[0, 1].item(), prediction[0, 0].item()) * 180 / np.pi
            if angle > 360:
                angle -= 360
            elif angle < 0:
                angle += 360
            pixels_angle_prediction = int(angle * pixels_per_degree)
            plt.plot([pixels_angle_prediction, pixels_angle_prediction], [0, height], 'r')

        plt.show()

def get_data_loaders(config, load_indices = False):
    # Load the data set from the CSV file:
    filenames, target_x, target_y, position_map, rotation_map, target_map, noisy_position_map, noisy_rotation_map, noisy_target_map =  \
        load_dataset_from_csv(os.path.join(config["dataset_folder"], config['csv_filename']))
    
    # Create and split the processed dataset
    image_folder = config["image_folder"]
    full_dataset = CustomDataset(image_folder=image_folder, image_names = filenames, target_x=target_x, target_y = target_y,  transform=transforms.ToTensor() \
                                 , augmentation=config["augment"])

    # check whether the pickle name exists:
    indices_pkl_name = os.path.join(config["dataset_folder"], "indices.pkl")
    if load_indices and not os.path.exists(indices_pkl_name):
        load_indices = False

    if load_indices:
        # after training, we want to load the same indices:
        with open(indices_pkl_name, "rb") as file:
            indices = pickle.load(file)
        train_indices = indices["train_indices"]
        val_indices = indices["val_indices"]
        test_indices = indices["test_indices"]
    else:
        dataset_size = len(full_dataset)
        train_size = int(config["split_ratio"] * len(full_dataset))
        test_size = len(full_dataset) - train_size
        val_size = int((1-config["split_ratio"]) * train_size)
        
        # Generate shuffled indices manually
        indices = torch.randperm(dataset_size).tolist()
        test_indices = indices[train_size:]
        val_indices = indices[:val_size]
        train_indices = indices[val_size:train_size]

        # pickle the indices to file:
        with open(os.path.join(config["dataset_folder"], "indices.pkl"), "wb") as file:
            pickle.dump({"train_indices": train_indices, "val_indices": val_indices, "test_indices": test_indices}, file)

    # Create subsets using indices
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    test_dataset = torch.utils.data.Subset(full_dataset, test_indices)

    train_loader = make_loader(train_dataset, shuffle=True, batch_size=config["batch_size"])
    val_loader = make_loader(val_dataset, shuffle=False, batch_size=config["batch_size_val"])
    test_loader = make_loader(test_dataset, shuffle=False, batch_size=config["batch_size_eval"])
    
    return train_loader, val_loader, test_loader, position_map, rotation_map, noisy_position_map, target_map, train_indices, val_indices, test_indices
