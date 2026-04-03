import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(2, 4, kernel_size=5, stride=4, padding=2)
        self.fc1 = nn.Linear(4 * 25 * 64, 2)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = torch.tanh(self.fc1(x))
        return x

class ImageBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.index = 0
        self.idx = 0

    def add(self, images, vectors):
        for image, vector in zip(images, vectors):
            if len(self.buffer) < self.capacity:
                self.buffer.append((image, vector))
            else:
                if self.index % 10 == 0: 
                    # Randomly replace an existing element
                    index = random.randint(self.idx, self.capacity - 1)
                    self.buffer[index] = (image, vector)
                    self.idx += 1
                else:
                    index = random.randint(self.idx, self.capacity - 1)
                    self.buffer[index] = (image, vector)

            self.index += 1

    def sample(self, size, augment_fn):
        samples = random.sample(self.buffer, min(size, len(self.buffer)))
        augmented_samples = [augment_fn(image, vector) for image, vector in samples]
        return augmented_samples
