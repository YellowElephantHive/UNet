import torch
import torch.nn as nn
from unet_blocks import DoubleConv
from unet_blocks import Down
from unet_blocks import OutConv


class UNet(nn.Module):
    def __init__(self, in_channel, n_classes):
        super(UNet, self).__init__()
        self.in_channel = in_channel
        self.n_classes = n_classes
        self.input_ch = DoubleConv(in_channel, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', 
                              align_corners=True)
        self.dconv1 = DoubleConv(1024 + 512, 512)
        self.dconv2 = DoubleConv(512 + 256, 256)
        self.dconv3 = DoubleConv(256 + 128, 128)
        self.dconv4 = DoubleConv(128 + 64, 64)
        self.output = OutConv(64, n_classes)
        
    def forward(self, x):
        conv1 = self.input_ch(x)
        conv2 = self.down1(conv1)
        conv3 = self.down2(conv2)
        conv4 = self.down3(conv3)
        conv5 = self.down4(conv4)
        up1 = self.up(conv5)
        concat = torch.cat([up1, conv4], dim=1) # [1024, 512]
        concat = self.dconv1(concat)
        up2 = self.up(concat)
        concat = torch.cat([up2, conv3], dim=1) # []
        concat = self.dconv2(concat)
        up3 = self.up(concat)
        concat = torch.cat([up3, conv2], dim=1)
        concat = self.dconv3(concat)
        up4 = self.up(concat)
        concat = torch.cat([up4, conv1], dim=1)
        concat = self.dconv4(concat)
        out = self.output(concat)
        return out