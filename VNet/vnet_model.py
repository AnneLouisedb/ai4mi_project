import torch
import torch.nn as nn
import torch.nn.functional as F

class VNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(VNet, self).__init__()

        # Downsampling path
        self.enc1 = self.encoder_block(in_channels, 16, use_pool=False)
        self.enc2 = self.encoder_block(16, 32)
        self.enc3 = self.encoder_block(32, 64)
        self.enc4 = self.encoder_block(64, 128)
        self.enc5 = self.encoder_block(128, 256)

        # Upsampling path
        self.dec4 = self.decoder_block(256, 128)
        self.dec3 = self.decoder_block(128, 64)
        self.dec2 = self.decoder_block(64, 32)
        self.dec1 = self.decoder_block(32, 16)

        # Final Convolution
        self.final_conv = nn.Conv3d(16, num_classes, kernel_size=1)

        # Initialize weights
        self.init_weights()

    def encoder_block(self, in_channels, out_channels, use_pool=True):
        layers = []
        if use_pool:
            # Check the depth of the input and skip depth pooling if it's too small
            layers.append(nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)))
            
        layers.extend([
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        ])
        return nn.Sequential(*layers)


    def decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Encoder
        if x.shape[2] > 1:  # Check depth dimension
            enc1 = self.enc1(x)             # [B, 16, D, H, W]
            enc2 = self.enc2(enc1)          # [B, 32, D/2, H/2, W/2]
        else:
            enc1 = self.enc1(x)
            enc2 = self.enc2[1:](enc1)  # Skip pooling on depth if needed

        enc3 = self.enc3(enc2)          # [B, 64, D/4, H/4, W/4]
        enc4 = self.enc4(enc3)          # [B, 128, D/8, H/8, W/8]
        enc5 = self.enc5(enc4)          # [B, 256, D/16, H/16, W/16]

        # Decoder with skip connections
        dec4 = self.dec4(enc5) + enc4   # [B, 128, D/8, H/8, W/8]
        dec3 = self.dec3(dec4) + enc3   # [B, 64, D/4, H/4, W/4]
        dec2 = self.dec2(dec3) + enc2   # [B, 32, D/2, H/2, W/2]
        dec1 = self.dec1(dec2) + enc1   # [B, 16, D, H, W]

        # Final Convolution
        out = self.final_conv(dec1)     # [B, num_classes, D, H, W]
        return out

