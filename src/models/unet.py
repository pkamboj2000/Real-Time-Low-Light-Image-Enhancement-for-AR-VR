"""
U-Net model implementation for low-light image enhancement.
Self-supervised training compatible with Noise2Noise approach.

Author: Pranjal Kamboj
Date: August 2025

This is my implementation of U-Net optimized for real-time low-light enhancement.
I've experimented with different architectures and this configuration works best
for AR/VR applications where speed and quality both matter.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    Double convolution block used in U-Net architecture.
    
    Standard building block: Conv2d -> BatchNorm -> ReLU -> Conv2d -> BatchNorm -> ReLU
    Works much better than single conv for this application.
    """
    
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        # Originally had bias=True but batch norm makes it redundant
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),  # inplace saves memory
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling with maxpool then double conv.
    
    Maxpool for downsampling - tried strided conv but maxpool works better
    for preserving important features in low-light images.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling then double conv.
    
    I experimented with both bilinear upsampling and transposed convolutions.
    Bilinear is faster and uses less memory, transposed conv gives slightly
    better quality. Made it configurable so you can choose.
    """

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            # Bilinear upsampling + conv to reduce channels
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # Learned upsampling with transposed conv
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatches - this happens sometimes with odd input sizes
        # Took me a while to figure this out when I first implemented U-Net!
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final output convolution - just a 1x1 conv to get the right number of channels."""
    
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net implementation for low-light image enhancement.
    
    Based on the original U-Net paper but modified for our specific use case.
    I've spent way too much time tweaking this architecture... 
    The skip connections are crucial for preserving fine details in dark images.
    
    Args:
        n_channels (int): Number of input channels (3 for RGB)
        n_classes (int): Number of output channels (3 for RGB)
        bilinear (bool): Use bilinear upsampling instead of transposed convolutions
        base_channels (int): Base number of channels (affects model size)
    """
    
    def __init__(self, n_channels=3, n_classes=3, bilinear=False, base_channels=64):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, base_channels)
        self.down1 = Down(base_channels, base_channels * 2)
        self.down2 = Down(base_channels * 2, base_channels * 4)
        self.down3 = Down(base_channels * 4, base_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_channels * 8, base_channels * 16 // factor)
        
        self.up1 = Up(base_channels * 16, base_channels * 8 // factor, bilinear)
        self.up2 = Up(base_channels * 8, base_channels * 4 // factor, bilinear)
        self.up3 = Up(base_channels * 4, base_channels * 2 // factor, bilinear)
        self.up4 = Up(base_channels * 2, base_channels, bilinear)
        self.outc = OutConv(base_channels, n_classes)

    def forward(self, x):
        # Encoder path - progressively downsample and increase channels
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)  # bottleneck layer
        
        # Decoder path - upsample and combine with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        # Apply sigmoid to ensure output is in [0, 1] range
        output = torch.sigmoid(logits)
        
        return output

    def count_parameters(self):
        """Count the number of trainable parameters. Useful for model size analysis."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightUNet(UNet):
    """
    Lightweight version of U-Net for mobile deployment.
    
    I created this after realizing the full U-Net was too heavy for real-time AR/VR.
    Uses half the channels and bilinear upsampling to reduce computation.
    """
    
    def __init__(self, n_channels=3, n_classes=3):
        # Use smaller base channels for mobile efficiency
        super().__init__(n_channels, n_classes, bilinear=True, base_channels=32)


class ResidualUNet(nn.Module):
    """
    U-Net with residual connections for better training stability.
    
    Still experimenting with this - the idea is to add the input image
    to the output for better gradients during training. Works well for
    image restoration tasks.
    """
    
    def __init__(self, n_channels=3, n_classes=3, base_channels=64):
        super().__init__()
        self.unet = UNet(n_channels, n_classes, base_channels=base_channels)
        
    def forward(self, x):
        enhanced = self.unet(x)
        # Residual connection: add input to enhancement
        return torch.clamp(x + enhanced, 0, 1)


def create_unet_model(model_type='standard', **kwargs):
    """
    Factory function to create different U-Net variants.
    
    Args:
        model_type (str): Type of model ('standard', 'lightweight', 'residual')
        **kwargs: Additional arguments for model initialization
    
    Returns:
        torch.nn.Module: U-Net model instance
    """
    if model_type == 'standard':
        return UNet(**kwargs)
    elif model_type == 'lightweight':
        return LightweightUNet(**kwargs)
    elif model_type == 'residual':
        return ResidualUNet(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test model creation and forward pass
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    model = create_unet_model('standard')
    model = model.to(device)
    
    # Test input
    x = torch.randn(1, 3, 256, 256).to(device)
    
    with torch.no_grad():
        output = model(x)
        
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    print(f"Device: {device}")
