import torch
import torch.nn as nn
from UNetModel.unet_blocks import (DownSample, DoubleConv, UpSample)
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.down_convolution_1 = DownSample(in_channels, 4,dilation = 1)
        self.down_convolution_2 = DownSample(4, 8,dilation = 2)
        self.down_convolution_3 = DownSample(8, 16,dilation = 4)
        self.down_convolution_4 = DownSample(16, 32,dilation = 8)

        self.bottle_neck = DoubleConv(32, 64,dilation = 16)

        self.up_convolution_1 = UpSample(64, 32,dilation = 8)
        self.up_convolution_2 = UpSample(32, 16,dilation = 4)
        self.up_convolution_3 = UpSample(16, 8,dilation = 2)
        self.up_convolution_4 = UpSample(8, 6,dilation = 1)

        # Adjust output to match the required number of output channels (6)
        self.out = nn.Conv1d(in_channels=6, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_convolution_1(x)
        down_2, p2 = self.down_convolution_2(p1)
        down_3, p3 = self.down_convolution_3(p2)
        down_4, p4 = self.down_convolution_4(p3)
        
        b = self.bottle_neck(p4)
        
        up_1 = self.up_convolution_1(b, down_4)
        up_2 = self.up_convolution_2(up_1, down_3)
        up_3 = self.up_convolution_3(up_2, down_2)
        up_4 = self.up_convolution_4(up_3, down_1)
        
        out = self.out(up_4)

        out = torch.sigmoid(out)
        return out
    
    def predict_masks(self,signal):
        self.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        in_sig = torch.tensor(signal,dtype=torch.float32).to(device)
        with torch.no_grad():
            region_masks = self(in_sig)

        return region_masks.cpu().numpy()