""" Full assembly of the parts to form the complete network """

from .unet_parts import *

# this function comes from ENet
def random_weights_init(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_normal_(m.weight.data)
        elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

class UNet(nn.Module): #(1, K) - one channel, K classes
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    # this is added 
     

    def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
            

class SUNet(nn.Module):  # (1, K) - one channel, K classes
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = SDoubleConv(n_channels, 16)
        self.down1 = Encoder_block(16, 32)
        self.down2 = Encoder_block(32, 64)
        self.down3 = Encoder_block(64, 128)
        self.down4 = Encoder_block(128, 256)

        factor = 2 if bilinear else 1

        # Bottleneck with dilated convolution
        self.bot = DilationBlock(256, 256)

        # Decoder blocks
        self.up1 = Decoder_block(256, 128, bilinear)
        self.up2 = Decoder_block(128, 64, bilinear)
        self.up3 = Decoder_block(64, 32, bilinear)
        self.up4 = Decoder_block(32, 16, bilinear)

        # Extra upsampling to ensure the output is the same size as the input
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        x0 = self.inc(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.bot(x4)

        # Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Apply the additional upsampling to match the input size
        x = self.final_upsample(x)

        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.bot = torch.utils.checkpoint(self.bot)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

    # Weight initialization method
    def init_weights(self, *args, **kwargs):
        self.apply(random_weights_init)