import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    """U-Net Block with two convolutions and optional batch normalization."""
    def __init__(self, in_channels, out_channels, batch_norm=True):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity()
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        return x


class Downsample(nn.Module):
    """Downsampling block with max pooling."""
    def __init__(self, in_channels, out_channels):
        super(Downsample, self).__init__()
        self.block = UNetBlock(in_channels, out_channels)

    def forward(self, x):
        return self.block(F.max_pool2d(x, kernel_size=2, stride=2))


class Upsample(nn.Module):
    """Upsampling block with concatenation."""
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.block = UNetBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)
        # Match dimensions
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.block(x)


class nnUNet(nn.Module):
    """nnU-Net model."""
    def __init__(self, n_channels, n_classes):
        super(nnUNet, self).__init__()
        self.encoder1 = UNetBlock(n_channels, 32)
        self.encoder2 = Downsample(32, 64)
        self.encoder3 = Downsample(64, 128)
        self.encoder4 = Downsample(128, 256)
        self.bottleneck = UNetBlock(256, 512)

        self.decoder4 = Upsample(512, 256)
        self.decoder3 = Upsample(256, 128)
        self.decoder2 = Upsample(128, 64)
        self.decoder1 = Upsample(64, 32)

        self.final_conv = nn.Conv2d(32, n_classes, kernel_size=1)

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        bottleneck = self.bottleneck(enc4)

        dec4 = self.decoder4(bottleneck, enc4)
        dec3 = self.decoder3(dec4, enc3)
        dec2 = self.decoder2(dec3, enc2)
        dec1 = self.decoder1(dec2, enc1)

        return self.final_conv(dec1)
