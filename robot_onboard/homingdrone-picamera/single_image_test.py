import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import torchvision.transforms as transforms
import torch.nn.init as init
import time
    
class CompactCNN_rgb(nn.Module):
    def __init__(self):
        super(CompactCNN_rgb, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=12, stride=4)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=4)
        self.fc = nn.Linear(in_features=2*1*7, out_features=2)

    def forward(self, x):
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 192x1800"
        x = torch.tanh(self.conv1(x))
        x = torch.tanh(self.conv2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 1*7*2)
        x = self.fc(x)
        return x

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)

class InceptionModule2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule2, self).__init__()
        self.branch5x5_1 = nn.Conv2d(in_channels, 4, kernel_size=5, stride=4, padding=2)
        self.branch3x3dbl_2 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.branch_dilated = nn.Conv2d(in_channels, 4, kernel_size=3, stride=4, padding=2, dilation=2)
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        branch5x5 = self.branch5x5_1(x)
        branch3x3dbl = self.branch3x3dbl_2(x)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=4, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_dilated = self.branch_dilated(x)
        outputs = torch.cat([branch5x5, branch3x3dbl, branch_pool, branch_dilated], dim=1)
        attention_map = self.spatial_attention(outputs)
        outputs = outputs * attention_map
        return outputs

class IncepAttentionCNN_rgb(nn.Module):
    def __init__(self):
        super(IncepAttentionCNN_rgb, self).__init__()
        self.inception1 = InceptionModule2(3)
        self.inception2 = InceptionModule2(14)
        self.conv3 = nn.Conv2d(in_channels=14, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=6, stride=1)
        self.fc = nn.Linear(in_features=4*1*52, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=2)
        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 192x1800"
        x = torch.tanh(self.inception1(x))
        x = torch.tanh(self.inception2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 52 * 1 * 4)
        x = torch.tanh(self.fc(x))
        x = self.fc2(x)
        return x

    

def predict(img_path, PATH, net_size):
    if net_size == 'small':
        net = CompactCNN_rgb()
    elif net_size == 'inceptattention':
        net = IncepAttentionCNN_rgb()

    net.load_state_dict(torch.load(PATH))

    image = Image.open(img_path)


    transform = transforms.Compose([
        transforms.Resize((192, 1800)),
        transforms.ToTensor()
    ])
    image = transform(image)
    image = image.unsqueeze(0)
    time_start = time.time()
    output = net(image)
    time_end = time.time()
    print(f"Inference time: {time_end - time_start} seconds")
    prediction = torch.atan2(output[0][1], output[0][0])
    prediction = prediction.item()

    distance = torch.norm(output[0])
    distance = distance.item()

    return prediction, distance