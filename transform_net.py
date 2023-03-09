import torch
import torch.nn as nn

class TransformNet(nn.Module):
    def __init__(self):
        super(TransformNet, self).__init__()
        self.ConvBlocks = nn.Sequential(
            ConvBlock(3, 32, 9, 1),
            nn.ReLU(),
            ConvBlock(32, 64, 3, 2),
            nn.ReLU(),
            ConvBlock(64, 128, 3, 2),
            nn.ReLU()
        )
        self.ResidualBlocks = nn.Sequential(
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3),
            ResidualBlock(128, 3)
        )
        self.UpSamplingBlocks = nn.Sequential(
            UpSamplingBlock(128, 64, 3, 2, 1),
            nn.ReLU(),
            UpSamplingBlock(64, 32, 3, 2, 1),
            nn.ReLU(),
            UpSamplingBlock(32, 3, 9, 1, norm = "None")
        )

    def forward(self, x):
        x = self.ConvBlocks(x)
        x = self.ResidualBlocks(x)
        out = self.UpSamplingBlocks(x)
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, norm = "instance"):
        super(ConvBlock, self).__init__()
        padding_size = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(padding_size)

        self.conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

        # Normalization Layers
        self.norm_type = norm
        if norm == "instance":
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine = True)
        elif norm == "batch":
            self.norm_layer = nn.BatchNorm2d(out_channels, affine = True)

    def forward(self, x):
        x = self.reflection_pad(x)
        x = self.conv_layer(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels = 128, kernel_size = 3):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock(channels, channels, kernel_size, stride = 1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvBlock(channels, channels, kernel_size, stride = 1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out 

class UpSamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, output_padding, norm="instance"):
        super(UpSamplingBlock, self).__init__()

        # Transposed Convolution
        padding_size = kernel_size // 2
        self.conv_transpose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding_size, output_padding)

        # Normalization Layers
        self.norm_type = norm
        if (norm == "instance"):
            self.norm_layer = nn.InstanceNorm2d(out_channels, affine = True)
        elif (norm == "batch"):
            self.norm_layer = nn.BatchNorm2d(out_channels, affine = True)

    def forward(self, x):
        x = self.conv_transpose(x)
        if self.norm_type == "None":
            out = x
        else:
            out = self.norm_layer(x)
        return out
