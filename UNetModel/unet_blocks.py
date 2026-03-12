import torch
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels,dilation = 1):
        super().__init__()
        self.conv_op = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding="same",dilation = dilation),
            nn.BatchNorm1d(out_channels),
            nn.PReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size=9, padding="same",dilation = dilation),
            nn.BatchNorm1d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv_op(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels,dilation = 1):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels,dilation)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        down = self.conv(x)
        p = self.pool(down)
        return down, p

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels,dilation = 1):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, (in_channels // 2), kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels,dilation)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Ensure that x1 and x2 are concatenated along the channel dimension
        #print(f"Shape of x1: {x1.shape}, Shape of x2: {x2.shape}")  # Debugging line
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)
