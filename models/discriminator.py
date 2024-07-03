import torch
import torch.nn as nn
import torch.nn.functional as F

from models.downsample import Downsample
from models.normalization import get_normalization_layer


class DiscriminatorBasicBlock(nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        do_downsample=True,
        do_instancenorm=True,
        normalization="in",
        parallelism=False,
    ):
        super().__init__()

        self.do_downsample = do_downsample
        self.do_instancenorm = do_instancenorm

        self.conv = nn.Conv2d(
            in_features, out_features, kernel_size=4, stride=1, padding=1
        )
        self.leakyrelu = nn.LeakyReLU(0.2, True)

        if do_instancenorm:
            self.instancenorm = get_normalization_layer(
                out_features, normalization=normalization, parallelism=parallelism,
            )

        if do_downsample:
            self.downsample = Downsample(out_features)

    def forward(self, x):
        x = self.conv(x)
        if self.do_instancenorm:
            x = self.instancenorm(x)
        x = self.leakyrelu(x)
        if self.do_downsample:
            x = self.downsample(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=64, avg_pooling=False):
        super().__init__()
        self.block1 = DiscriminatorBasicBlock(
            in_channels,
            features,
            do_downsample=True,
            do_instancenorm=False,
        )
        self.block2 = DiscriminatorBasicBlock(
            features,
            features * 2,
            do_downsample=True,
            do_instancenorm=True,
        )
        self.block3 = DiscriminatorBasicBlock(
            features * 2,
            features * 4,
            do_downsample=True,
            do_instancenorm=True,
        )
        self.block4 = DiscriminatorBasicBlock(
            features * 4,
            features * 8,
            do_downsample=False,
            do_instancenorm=True,
        )
        self.conv = nn.Conv2d(
            features * 8,
            1,
            kernel_size=4,
            stride=1,
            padding=1,
        )
        self.avg_pooling = avg_pooling

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        if self.avg_pooling:
            x = F.avg_pool2d(x, x.size()[2:])
            x = torch.flatten(x, 1)
        return x

    def set_requires_grad(self, requires_grad=False):
        for param in self.parameters():
            param.requires_grad = requires_grad


if __name__ == "__main__":
    x = torch.randn((5, 3, 256, 256))
    print(x.shape)
    model = Discriminator(in_channels=3, avg_pooling=True)
    preds = model(x)
    print(preds.shape)
