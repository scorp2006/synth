"""
CycleGAN Generator Network Architecture
Compatible with standard CycleGAN pretrained weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block with instance normalization"""
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, 3),
            nn.InstanceNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    CycleGAN Generator (ResNet-based)
    Standard architecture compatible with most pretrained weights
    """
    def __init__(self, input_nc=3, output_nc=3, n_residual_blocks=9, ngf=64):
        super(Generator, self).__init__()

        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, 7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = ngf
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2

        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, 7),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class GeneratorUNet(nn.Module):
    """
    U-Net based Generator (alternative architecture)
    Used by some weather synthesis models
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64):
        super(GeneratorUNet, self).__init__()

        # Encoder
        self.enc1 = self._encoder_block(input_nc, ngf, normalize=False)
        self.enc2 = self._encoder_block(ngf, ngf * 2)
        self.enc3 = self._encoder_block(ngf * 2, ngf * 4)
        self.enc4 = self._encoder_block(ngf * 4, ngf * 8)
        self.enc5 = self._encoder_block(ngf * 8, ngf * 8)
        self.enc6 = self._encoder_block(ngf * 8, ngf * 8)
        self.enc7 = self._encoder_block(ngf * 8, ngf * 8)
        self.enc8 = self._encoder_block(ngf * 8, ngf * 8, normalize=False)

        # Decoder
        self.dec1 = self._decoder_block(ngf * 8, ngf * 8, dropout=True)
        self.dec2 = self._decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.dec3 = self._decoder_block(ngf * 16, ngf * 8, dropout=True)
        self.dec4 = self._decoder_block(ngf * 16, ngf * 8)
        self.dec5 = self._decoder_block(ngf * 16, ngf * 4)
        self.dec6 = self._decoder_block(ngf * 8, ngf * 2)
        self.dec7 = self._decoder_block(ngf * 4, ngf)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, output_nc, 4, 2, 1),
            nn.Tanh()
        )

    def _encoder_block(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)

    def _decoder_block(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        e6 = self.enc6(e5)
        e7 = self.enc7(e6)
        e8 = self.enc8(e7)

        # Decoder with skip connections
        d1 = self.dec1(e8)
        d2 = self.dec2(torch.cat([d1, e7], 1))
        d3 = self.dec3(torch.cat([d2, e6], 1))
        d4 = self.dec4(torch.cat([d3, e5], 1))
        d5 = self.dec5(torch.cat([d4, e4], 1))
        d6 = self.dec6(torch.cat([d5, e3], 1))
        d7 = self.dec7(torch.cat([d6, e2], 1))

        return self.final(torch.cat([d7, e1], 1))


def load_generator(model_path, device='cuda', arch='resnet'):
    """
    Load a pretrained generator

    Args:
        model_path: Path to .pth weights file
        device: 'cuda' or 'cpu'
        arch: 'resnet' (CycleGAN) or 'unet'

    Returns:
        Loaded generator model
    """
    if arch == 'resnet':
        model = Generator(input_nc=3, output_nc=3, n_residual_blocks=9)
    else:
        model = GeneratorUNet(input_nc=3, output_nc=3)

    # Load state dict
    state_dict = torch.load(model_path, map_location=device)

    # Handle different checkpoint formats
    if 'model' in state_dict:
        state_dict = state_dict['model']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif 'netG_A' in state_dict:
        state_dict = state_dict['netG_A']
    elif 'netG' in state_dict:
        state_dict = state_dict['netG']

    # Remove 'module.' prefix if present (from DataParallel)
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model
