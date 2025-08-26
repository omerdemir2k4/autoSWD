import torch
import torch.nn as nn

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_prob=0.15):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.dropout = nn.Dropout(p=dropout_prob)  # Dropout eklendi

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1, padding="same")
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)  # Dropout conv1 ile conv2 arasında
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class AttentionBlock(nn.Module):
    def __init__(self, g, x, inter_channels, dropout_prob=0.1):
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
            nn.Conv1d(inter_channels, 1, kernel_size=1, stride=1, padding="same", bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(p=dropout_prob)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.w_g(g)
        x1 = self.w_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        psi = self.dropout(psi)  # Dropout attention mask

        return psi * x


class AttentionResUNet(nn.Module):
    def __init__(self, n_classes=1, in_channels=1, out_channels=1):
        super().__init__()

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv1 = ResidualBlock1D(in_channels=in_channels, out_channels=32)
        self.conv2 = ResidualBlock1D(in_channels=32, out_channels=64)
        self.conv3 = ResidualBlock1D(in_channels=64, out_channels=128)
        self.conv4 = ResidualBlock1D(in_channels=128, out_channels=256)
        self.conv5 = ResidualBlock1D(in_channels=256, out_channels=512)

        self.up5 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.att5 = AttentionBlock(g=256, x=256, inter_channels=128)
        self.upconv5 = ResidualBlock1D(in_channels=512, out_channels=256)

        self.up4 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.att4 = AttentionBlock(g=128, x=128, inter_channels=64)
        self.upconv4 = ResidualBlock1D(in_channels=256, out_channels=128)

        self.up3 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.att3 = AttentionBlock(g=64, x=64, inter_channels=32)
        self.upconv3 = ResidualBlock1D(in_channels=128, out_channels=64)

        self.up2 = nn.ConvTranspose1d(in_channels=64, out_channels=32, kernel_size=2, stride=2)
        self.att2 = AttentionBlock(g=32, x=32, inter_channels=16)
        self.upconv2 = ResidualBlock1D(in_channels=64, out_channels=32)

        self.final_conv = nn.Conv1d(32, out_channels, kernel_size=1, stride=1, padding="same")

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
