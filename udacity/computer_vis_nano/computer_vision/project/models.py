import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super().__init__()
        
        # TODO: Define all the layers of this CNN, the only requirements are:
        # 1. This network takes in a square (same width and height), greyscale
        #     image as input
        # 2. It ends with a linear layer that represents the keypoints
        # it's suggested that you make this last layer output 136 values, 2
        # for each of the 68 keypoint (x, y) pairs
        # As an example, you've been given a convolutional layer, which you may
        # (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        # Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and
        # other layers (such as dropout or batch normalization) to avoid
        # overfitting
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop10 = nn.Dropout(p=0.1)
        self.drop20 = nn.Dropout(p=0.2)
        self.drop30 = nn.Dropout(p=0.3)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(128, 136)
        self.bn4 = nn.BatchNorm1d(136)
            
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.drop10(x)
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.drop10(x)
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.drop20(x)
        #x = x.view(x.size(0), -1)
        #x = x.view(x, 1)
        x = torch.flatten(self.gap(x), 1)
        x = F.relu(self.bn4(self.fc1(x)))
        x = self.drop30(x)
        return x


# Cleaner
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size,
                stride=stride,  # stride handles down-sampling in place of pool
                padding=padding,
                bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


class ResidBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.skip = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.main(x)
        skip = self.skip(x)
        return self.relu(out + skip)

    
class Net2(nn.Module):
    def __init__(self, n_classes=136):
        super().__init__()
        self.stem = ConvBlock(1, 32)
        self.stage1 = ResidBlock(32, 64, stride=2)
        self.stage2 = ResidBlock(64, 128, stride=2)
        self.stage3 = ResidBlock(128, 256, stride=2)
        self.stage4 = ResidBlock(256, 512, stride=2)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            ###
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            ###
            nn.Linear(512, n_classes))
        self.apply(init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    
def init_weights(m):
    if  isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, ResidBlock):
            nn.init.zeros_(m.conv2[1].weight)
        


def count_features(mod, input_shape=(1, 1, 64, 64)):
    with torch.no_gra():
        x = torch.zeros(*input_shape)
        y = mod.stem(x)
        y = mod.stage1(y)
        y = mod.stage2(y)
        y = mod.stage3(y)
        print('Feature map:', y.shape)


def verify_init(mod):
    for name, p in mod.named_parameters():
        if p.requires_grad and 'weight' in name:
            # should have mean near 0, std on roughly [0.02, 0.2]
            print(name, p.mean().item(), p.std().item())


def seed_everything(seed: int=1234):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id):
    worker_seed = torch.inital_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
