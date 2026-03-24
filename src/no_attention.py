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
        """Perform forward propagation."""
        return self.conv(x)


class UpBlock(nn.Module):
    """Upsampling convolutional block."""

    def __init__(self, in_channel: int, skip_channel: int, out_channel: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channel, in_channel // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channel // 2 + skip_channel, out_channel)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        """Perform forward propagation."""
        up = self.up(x)

        if up.shape[2:] != skip.shape[2:]:
            up = f.interpolate(up, size=skip.shape[2:], mode="bilinear", align_corners=False)

        return self.conv(torch.cat([skip, up], dim=1))


class AdhdUnet(nn.Module):
    """U-Net implementation without attention."""

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

        self.conv = ConvBlock(128, 64)

        self.final_up = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, classes, kernel_size=1)

        self._initialize_decoder_weights()

    def forward(self, x: Tensor) -> Tensor:
        """Perform forward propagation."""
        e0 = self.encoder0(x)
        e1 = self.encoder1(self.pool(e0))
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = self.up4(e4, e3)
        d3 = self.up3(d4, e2)
        d2 = self.up2(d3, e1)
        d2_up = self.up1(d2)

        if d2_up.shape[2:] != e0.shape[2:]:
            d2_up = f.interpolate(d2_up, size=e0.shape[2:], mode="bilinear", align_corners=False)

        d1 = self.conv(torch.cat([e0, d2_up], dim=1))

        out = self.final_up(d1)
        if out.shape[2:] != x.shape[2:]:
            out = f.interpolate(out, size=x.shape[2:], mode="bilinear", align_corners=False)

        return self.final_conv(out)

    def _initialize_decoder_weights(self) -> None:
        self.up4.apply(_init_weights)
        self.up3.apply(_init_weights)
        self.up2.apply(_init_weights)
        self.up1.apply(_init_weights)
        self.conv.apply(_init_weights)
        self.final_up.apply(_init_weights)
        self.final_conv.apply(_init_weights)
