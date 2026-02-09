import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import numpy as np
from tqdm import tqdm

class EEGDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

    
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same") if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)
  

class AttentionBlock(nn.Module):
    def __init__(self, g, x, inter_channels):
        super().__init__()
        
        self.w_g = nn.Sequential(
            nn.Conv1d(g, inter_channels, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm1d(inter_channels)
            )
        
        self.w_x = nn.Sequential(
            nn.Conv1d(x, inter_channels, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm1d(inter_channels)
            )
        
        self.psi = nn.Sequential(
            nn.Conv1d(inter_channels, 1, kernel_size=1, stride=1, padding="same",  bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)
        
        return psi*x


class AttentionResUNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=1, out_channels=1):
        super().__init__() 
        
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv1 = ResidualBlock1D(in_channels=in_channels, out_channels=64)
        self.conv2 = ResidualBlock1D(in_channels=64, out_channels=128)
        self.conv3 = ResidualBlock1D(in_channels=128, out_channels=256)
        self.conv4 = ResidualBlock1D(in_channels=256, out_channels=512)
        self.conv5 = ResidualBlock1D(in_channels=512, out_channels=1024)
        
        self.up5 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(g=512, x=512, inter_channels=256)
        self.upconv5 = ResidualBlock1D(in_channels=1024, out_channels=512)
        
        self.up4 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(g=256, x=256, inter_channels=128)
        self.upconv4 = ResidualBlock1D(in_channels=512, out_channels=256)
        
        self.up3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(g=128, x=128, inter_channels=64)
        self.upconv3 = ResidualBlock1D(in_channels=256, out_channels=128)
        
        self.up2 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(g=64, x=64, inter_channels=32)
        self.upconv2 = ResidualBlock1D(in_channels=128, out_channels=64)
        
        self.final_conv = nn.Conv1d(64, out_channels, kernel_size=1, stride=1, padding="same")
        
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.maxpool(x1)
        x2 = self.conv2(x2)
        x3 = self.maxpool(x2)
        x3 = self.conv3(x3)
        x4 = self.maxpool(x3)
        x4 = self.conv4(x4)
        x5 = self.maxpool(x4)
        x5 = self.conv5(x5)
        
        d5 = self.up5(x5)
        x4 = self.att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.upconv5(d5)
        
        d4 = self.up4(d5)
        x3 = self.att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.upconv4(d4)
        
        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.upconv3(d3)
        
        d2 = self.up2(d3)
        x1 = self.att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.upconv2(d2)
        
        d1 = self.final_conv(d2)
        
        return torch.sigmoid(d1)
    

class DS_AttentionResUNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=1, out_channels=1):
        super().__init__()

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1 = ResidualBlock1D(in_channels, 64)
        self.conv2 = ResidualBlock1D(64, 128)
        self.conv3 = ResidualBlock1D(128, 256)
        self.conv4 = ResidualBlock1D(256, 512)
        self.conv5 = ResidualBlock1D(512, 1024)

        self.up5 = nn.ConvTranspose1d(1024, 512, 2, 2)
        self.att5 = AttentionBlock(512, 512, 256)
        self.upconv5 = ResidualBlock1D(1024, 512)

        self.up4 = nn.ConvTranspose1d(512, 256, 2, 2)
        self.att4 = AttentionBlock(256, 256, 128)
        self.upconv4 = ResidualBlock1D(512, 256)

        self.up3 = nn.ConvTranspose1d(256, 128, 2, 2)
        self.att3 = AttentionBlock(128, 128, 64)
        self.upconv3 = ResidualBlock1D(256, 128)

        self.up2 = nn.ConvTranspose1d(128, 64, 2, 2)
        self.att2 = AttentionBlock(64, 64, 32)
        self.upconv2 = ResidualBlock1D(128, 64)

        self.final_conv = nn.Conv1d(64, out_channels, 1, 1, padding="same")

        # =¡ Deep supervision outputs
        self.ds3 = nn.Conv1d(128, out_channels, 1)
        self.ds4 = nn.Conv1d(256, out_channels, 1)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        d5 = self.upconv5(torch.cat((self.att5(self.up5(x5), x4), self.up5(x5)), dim=1))
        d4 = self.upconv4(torch.cat((self.att4(self.up4(d5), x3), self.up4(d5)), dim=1))
        d3 = self.upconv3(torch.cat((self.att3(self.up3(d4), x2), self.up3(d4)), dim=1))
        d2 = self.upconv2(torch.cat((self.att2(self.up2(d3), x1), self.up2(d3)), dim=1))

        out_main = torch.sigmoid(self.final_conv(d2))
        out_ds3 = torch.sigmoid(self.ds3(d3))  # Daha yukar1da, daha kaba seviye
        out_ds4 = torch.sigmoid(self.ds4(d4))  # Daha da yukar1

        if self.training:
            return out_main, out_ds3, out_ds4
        else:
            return out_main
