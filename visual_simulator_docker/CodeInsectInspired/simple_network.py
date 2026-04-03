import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self, n_outputs = 2, n_input_channels=3, n_hiddens = 100, n_channels_1 = 10, n_channels_2 = 10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(n_input_channels, n_channels_1, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(n_channels_1, n_channels_2, kernel_size=5, stride=4, padding=2)
        self.fc1 = nn.LazyLinear(n_hiddens) 
        self.af1 = nn.ReLU()
        self.fc2 = nn.Linear(n_hiddens, n_outputs)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.af1(x)
        x = self.fc2(x)
        
        return x
    
class CompactCNN_rgb(nn.Module):
    def __init__(self, n_outputs = 2, n_input_channels=3):
        super(CompactCNN_rgb, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=n_input_channels, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=12, stride=4)
        self.conv4 = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=1, stride=4)
        self.fc = nn.Linear(in_features=2*1*7, out_features=n_outputs)

    def forward(self, x):
        x = F.interpolate(x, size=(192, 1800), mode='bilinear', align_corners=False)
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 1x192x1800"
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
        # Aggregate channel information using average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)  # Average pooling
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # Max pooling
        # Concatenate the pooled features
        x = torch.cat([avg_out, max_out], dim=1)
        # Apply convolution to generate spatial attention map
        x = self.conv(x)
        # Apply sigmoid to get attention weights
        return self.sigmoid(x)

class InceptionModule2(nn.Module):
    def __init__(self, in_channels):
        super(InceptionModule2, self).__init__()
        self.branch5x5_1 = nn.Conv2d(in_channels, 4, kernel_size=5, stride=4, padding=2)
        self.branch3x3dbl_2 = nn.Conv2d(in_channels, 4, kernel_size=3, stride=2, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1)
        self.branch_pool = nn.Conv2d(in_channels, 2, kernel_size=1)
        self.branch_dilated = nn.Conv2d(in_channels, 4, kernel_size=3, stride=4, padding=2, dilation=2)
        
        # Add spatial attention module
        self.spatial_attention = SpatialAttention()

    def forward(self, x):
        branch5x5 = self.branch5x5_1(x)
        branch3x3dbl = self.branch3x3dbl_2(x)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=4, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        branch_dilated = self.branch_dilated(x)

        # Concatenate all branches
        outputs = torch.cat([branch5x5, branch3x3dbl, branch_pool, branch_dilated], dim=1)
        
        # Apply spatial attention
        attention_map = self.spatial_attention(outputs)
        # Multiply the attention map with the concatenated outputs
        outputs = outputs * attention_map

        return outputs


class IncepAttentionCNN_rgb(nn.Module):
    def __init__(self, n_outputs=2, n_input_channels=3):
        super(IncepAttentionCNN_rgb, self).__init__()
        self.inception1 = InceptionModule2(n_input_channels)
        self.inception2 = InceptionModule2(14)  # Adjust the input channels based on the output of inception1
        self.conv3 = nn.Conv2d(in_channels=14, out_channels=8, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(in_channels=8, out_channels=4, kernel_size=6, stride=1)
        self.fc = nn.Linear(in_features=4*1*52, out_features=16)
        self.fc2 = nn.Linear(in_features=16, out_features=n_outputs)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # resize the input to 192x1800
        x = F.interpolate(x, size=(192, 1800), mode='bilinear', align_corners=False)
        assert x.size()[2:] == (192, 1800), "Input dimensions must be 192x1800"
        x = torch.tanh(self.inception1(x))
        x = torch.tanh(self.inception2(x))
        x = torch.tanh(self.conv3(x))
        x = torch.tanh(self.conv4(x))
        x = x.view(-1, 52 * 1 * 4)  # Adjust according to the actual output size
        x = torch.tanh(self.fc(x))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SimpleCNN()
    print(model)
    # Initialize lazy layers by passing a dummy input
    dummy_input = torch.randn(1, 3, 400, 1024)  # Adjust dimensions as needed
    model(dummy_input)
    print("Number of parameters:", count_parameters(model))
    
    model = CompactCNN_rgb()
    print(model)
    print("Number of parameters:", count_parameters(model))

    model = IncepAttentionCNN_rgb()
    print(model)
    print("Number of parameters:", count_parameters(model))