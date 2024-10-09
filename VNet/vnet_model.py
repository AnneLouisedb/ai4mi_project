import torch
import torch.nn as nn
import torch.nn.functional as F


class VNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNet, self).__init__()
        self.enc1 = self.down_block(in_channels, 16)
        self.enc2 = self.down_block(16, 32)
        self.enc3 = self.down_block(32, 64)
        self.enc4 = self.down_block(64, 128)

        self.middle = self.double_conv(128, 256)

        self.dec4 = self.up_block(256, 128)
        self.dec3 = self.up_block(128, 64)
        self.dec2 = self.up_block(64, 32)
        self.dec1 = self.up_block(32, 16)

        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def down_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool3d(kernel_size=2),
            self.double_conv(in_channels, out_channels)
        )

    def up_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            self.double_conv(out_channels, out_channels)
        )

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)

        middle = self.middle(enc4)

        dec4 = self.dec4(middle)
        dec3 = self.dec3(dec4 + enc3)  # Skip connection
        dec2 = self.dec2(dec3 + enc2)  # Skip connection
        dec1 = self.dec1(dec2 + enc1)  # Skip connection

        return self.final_conv(dec1)
