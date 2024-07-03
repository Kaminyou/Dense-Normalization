import torch.nn as nn


class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.reflectionpad = nn.ReflectionPad2d(1)
        self.conv = nn.Conv2d(features, features, kernel_size=3, stride=2)

    def forward(self, x):
        x = self.reflectionpad(x)
        x = self.conv(x)
        return x
