import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from PIL import Image



class GazeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        # image = Image.open(img_path).convert('L') 
        image = Image.open(img_path)
        numeric_data = self.annotations.iloc[index, 4:6].values.astype(float)
        y_label = torch.tensor(numeric_data, dtype=torch.float) 

        if self.transform:
            image = self.transform(image)
        
            
        return (image, y_label)