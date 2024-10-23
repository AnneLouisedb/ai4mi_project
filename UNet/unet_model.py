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
        self.decoder = nn.Identity()  # Placeholder
        self.deep_supervision = False #deep_supervision
        
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
            

class UNetDR(nn.Module):  # (1, K) - one channel, K classes
    def __init__(self, n_channels, n_classes, bilinear=False, deep_supervision=False):
        super(UNetDR, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.decoder = nn.Identity()  # Placeholder
        self.deep_supervision = False #deep_supervision

        self.outnew1 = OutConv(128, 5)
        self.outnew2 = OutConv(64, 5)
        self.outnew3 = OutConv(32, 5)


        self.inc = SDoubleConv(n_channels, 16)
        self.down1 = Encoder_block(16, 32)
        self.down2 = Encoder_block(32, 64)
        self.down3 = Encoder_block(64, 128)
        self.down4 = Encoder_block(128, 256)

        
        # Bottleneck with dilated convolution
        self.bot = DilationBlock(256, 256)

        # Decoder blocks
        self.up1 = Decoder_block(256, 128)
        self.up2 = Decoder_block(128, 64)
        self.up3 = Decoder_block(64, 32)
        self.up4 = Decoder_block(32, 16)

        # Extra upsampling to ensure the output is the same size as the input
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # Final output layer
        self.outc = OutConv(16, n_classes)

    def set_deep_supervision(self, enabled):
        self.deep_supervision = enabled

  
    def forward(self, x):
        # print("DEEP SUPERVISION?", self.deep_supervision)
        x0 = self.inc(x)
        
        x1 = self.down1(x0)
       
        x2 = self.down2(x1)
      
        x3 = self.down3(x2)
        
        x4 = self.down4(x3)
       

        x5 = self.bot(x4)

        #Decoder path
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # Apply the additional upsampling to match the input size
        x = self.final_upsample(x)

        # output = self.outc(x)
        # return output

        x = self.up1(x5, x4)

        x = x.to(torch.float32)
        ds4 =  self.outnew1(x)

        x = self.up2(x, x3)
        ds3 = self.outnew2(x)

        x = self.up3(x, x2)
        ds2 = self.outnew3(x) 
        x = self.up4(x, x1)
       
        ds1 = self.outc(self.final_upsample(x)) 

        if self.deep_supervision:
            return [ds1, ds2, ds3, ds4] # output is a tensor, but it should be a list
        else:
            return ds1


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


# Shallow UNet

class SUNet(nn.Module): #(1, K) - one channel, K classes
    def __init__(self, n_channels, n_classes, dropout_prob):
        super(SUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        # self.bilinear = bilinear
        self.decoder = nn.Identity()  # Placeholder
        self.deep_supervision = False #deep_supervision
        
        self.inc = (DoubleConv(n_channels, 64)) # first layer
        self.down1 = (Down(64, 128, dropout_prob=dropout_prob)) # second layer
        self.down2 = (Down(128, 256,dropout_prob= dropout_prob)) # third laywer

        factor = 2 
     
        self.down3 = (Down(256, 512 // factor, dropout_prob=dropout_prob)) # fourth fourth layer - ensure the same size in the max pooling
      
       
        self.up1 = (UpS(512, 256  // factor, dropout_prob)) #  [256, 512, 3, 3], expected input[8, 768, 64, 64] to have 512 channels, but got 768 channels instead
     
        self.up2 = (UpS(256, 128 // factor,  dropout_prob))
        
        self.up3 = (UpS(128, 64 ,  dropout_prob))
       
        self.outc = (OutConv(64, n_classes))
      


    def forward(self, x):
        x1 = self.inc(x)
      
        x2 = self.down1(x1)
       
        x3 = self.down2(x2)
        
        x4 = self.down3(x3)
       
        x = self.up1(x4, x3)
        
        x = self.up2(x, x2)
    
        x = self.up3(x, x1)
        
        logits = self.outc(x)
        
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
     
        self.outc = torch.utils.checkpoint(self.outc)


     

    def init_weights(self, *args, **kwargs):
                self.apply(random_weights_init)
