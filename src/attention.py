import torch
import torch.nn.functional as f
from torch import Tensor, nn
from torchvision.models import ResNet34_Weights, resnet34


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


class AttentionGate(nn.Module):
    """Module for calculating additive soft attention."""

    def __init__(self, g_channels: int, l_channels: int, int_channels: int) -> None:
        super().__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(g_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels),
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(l_channels, int_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(int_channels),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(int_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

    def forward(self, g: Tensor, x: Tensor) -> Tensor:
        """Perform forwared propagation."""
        g1 = self.w_g(g)
        x1 = self.w_x(x)

        if g1.shape[2:] != x1.shape[2:]:
            g1 = f.interpolate(g1, size=x1.shape[2:], mode="bilinear", align_corners=False)

        psi = f.relu(g1 + x1, inplace=True)
        alpha = self.psi(psi)
        return x * alpha


class ConvBlock(nn.Module):
    """Standard convolutional block from U-Net."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Perform forwared propagation."""
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling convolutional block with attention gate."""

    def __init__(self, in_channel: int, skip_channel: int, out_channel: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.attention = AttentionGate(
            g_channels=in_channel // 2, l_channels=skip_channel, int_channels=in_channel // 4
        )
        self.conv = ConvBlock(in_channel // 2 + skip_channel, out_channel)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Perform forwared propagation."""
        up = self.up(x)

        if up.shape[2:] != skip.shape[2:]:
            up = f.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)

        attention = self.attention(g=up, x=skip)
        return self.conv(torch.cat([attention, up], dim=1))


class AttentionUnet(nn.Module):
    """U-Net implementation with attention."""

    def __init__(self, in_channels: int = 1, classes: int = 3) -> None:
        super().__init__()
        encoder = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)

        self.encoder0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)

        original_weights = self.encoder0[0].weight.data
        self.encoder0[0] = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder0[0].weight.data = original_weights.mean(dim=1, keepdim=True)

        self.pool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.up4 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128, 64, 64)
        self.up1 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)

        self.attention = AttentionGate(g_channels=64, l_channels=64, int_channels=32)
        self.conv = ConvBlock(128, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        self._initialize_decoder_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Perform forwared propagation."""
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d2_up = self.up1(d2)

        d1_att = self.attention(g=d2_up, x=e0)
        d1 = self.conv(torch.cat([d1_att, d2_up], dim=1))

        out = self.final_up(d1)
        if out.shape[2:] != x.shape[2:]:
            out = f.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.final_conv(out)

    def _initialize_decoder_weights(self) -> None:
        self.up4.apply(_init_weights)
        self.up3.apply(_init_weights)
        self.up2.apply(_init_weights)
        self.up1.apply(_init_weights)
        self.attention.apply(_init_weights)
        self.conv.apply(_init_weights)
        self.final_up.apply(_init_weights)
        self.final_conv.apply(_init_weights)
