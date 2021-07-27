import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.double_conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(in_channel, out_channel))
        
    def forward(self, x):
        return self.maxpool_conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(OutConv, self).__init__()
        self.out_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        
    def forward(self, x):
        return self.out_conv(x)
