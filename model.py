import torch

class DoubleConv(torch.nn.Module):
    """
    Helper Class which implements the intermediate Convolutions
    """
    def __init__(self, in_channels, out_channels):
        
        super().__init__()
        self.step = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU(),
                                        torch.nn.Conv2d(out_channels, out_channels, 3, padding=1),
                                        torch.nn.ReLU())
        
    def forward(self, X):
        return self.step(X)

class UNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = DoubleConv(3, 64)
        self.layer2 = DoubleConv(64, 128)
        self.layer3 = DoubleConv(128, 256)
        self.layer4 = DoubleConv(256, 512)
        
        self.layer5 = DoubleConv(512+256, 256)
        self.layer6 = DoubleConv(256+128, 128)
        self.layer7 = DoubleConv(128+64, 64)
        self.layer8 = torch.nn.Conv2d(64, 1, 1)
        
        self.maxpool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        
        x1 = self.layer1(x)
        x1m = self.maxpool(x1)
        
        x2 = self.layer2(x1m)
        x2m = self.maxpool(x2)
        
        x3 = self.layer3(x2m)
        x3m = self.maxpool(x3)
        
        x4 = self.layer4(x3m)
        
        x5 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x4)
        x5 = torch.cat([x5, x3], dim=1)
        x5 = self.layer5(x5)
        
        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x6 = self.layer6(x6)
        
        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x1], dim=1)
        x7 = self.layer7(x7)
        
        ret = self.layer8(x7)
        return ret

class UNet_Full(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.e_layer1 = DoubleConv(3, 64)
        self.e_layer2 = DoubleConv(64, 128)
        self.e_layer3 = DoubleConv(128, 256)
        self.e_layer4 = DoubleConv(256, 512)
        self.e_layer5 = DoubleConv(512, 1024)
        
        # Decoder
        self.d_layer6 = DoubleConv(1024+512, 512)
        self.d_layer7 = DoubleConv(512+256, 256)
        self.d_layer8 = DoubleConv(256+128, 128)
        self.d_layer9 = DoubleConv(128+64, 64)
        self.d_layer10 = torch.nn.Conv2d(64, 1, 1)
        
        self.maxpool = torch.nn.MaxPool2d(2)
        
    def forward(self, x):
        
        x1 = self.e_layer1(x)
        x1m = self.maxpool(x1)
        
        x2 = self.e_layer2(x1m)
        x2m = self.maxpool(x2)
        
        x3 = self.e_layer3(x2m)
        x3m = self.maxpool(x3)
        
        x4 = self.e_layer4(x3m)
        x4m = self.maxpool(x4)
        
        x5 = self.e_layer5(x4m)
        
        x6 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x5)
        x6 = torch.cat([x6, x4], dim=1)
        x6 = self.d_layer6(x6)
        
        x7 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x6)
        x7 = torch.cat([x7, x3], dim=1)
        x7 = self.d_layer7(x7)

        x8 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x7)
        x8 = torch.cat([x8, x2], dim=1)
        x8 = self.d_layer8(x8)

        x9 = torch.nn.Upsample(scale_factor=2, mode="bilinear")(x8)
        x9 = torch.cat([x9, x1], dim=1)
        x9 = self.d_layer9(x9)
        
        ret = self.d_layer10(x9)
        
        return ret