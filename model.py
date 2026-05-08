"""
Model architecture for Depth-Aware Pseudo-Segmentation.
Implements a multi-head U-Net model with class segmentation and depth estimation branches.
"""

import logging
from typing import Tuple, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import config

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """Convolutional block with double convolution and optional batch normalization."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
        batch_norm: bool = True,
    ) -> None:
        """
        Initialize conv block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Convolution kernel size
            padding: Padding size
            batch_norm: Whether to use batch normalization
        """
        super().__init__()
        layers = []
        
        # First convolution
        layers.append(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        # Second convolution
        layers.append(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=not batch_norm,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.block(x)


class UpBlock(nn.Module):
    """Upsampling block with skip connection."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        skip_channels: int = 0,
    ) -> None:
        """
        Initialize upsampling block.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            skip_channels: Number of skip connection channels
        """
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = ConvBlock(
            in_channels + skip_channels,
            out_channels,
        )
    
    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with optional skip connection."""
        x = self.upsample(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DepthAwareUNet(nn.Module):
    """
    Multi-head U-Net model for semantic segmentation with pseudo-depth estimation.
    
    Architecture:
    - Encoder: ResNet backbone (optional) or custom encoder
    - Decoder: Symmetric decoder with skip connections
    - Segmentation Head: Predicts class labels (num_classes)
    - Depth Head: Predicts pseudo-depth map (1 channel)
    
    The model uses spatial priors from vertical position to improve segmentation,
    especially for distant objects.
    """
    
    def __init__(
        self,
        num_classes: int = config.NUM_CLASSES,
        encoder_depth: int = config.ENCODER_DEPTH,
        decoder_channels: List[int] = config.DECODER_CHANNELS,
        use_pretrained_backbone: bool = config.PRETRAINED,
        freeze_backbone: bool = config.FREEZE_BACKBONE,
    ) -> None:
        """
        Initialize the model.
        
        Args:
            num_classes: Number of segmentation classes
            encoder_depth: Depth of encoder layers
            decoder_channels: List of decoder channel dimensions
            use_pretrained_backbone: Whether to use pretrained ResNet
            freeze_backbone: Whether to freeze backbone layers
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.encoder_depth = encoder_depth
        self.decoder_channels = decoder_channels
        
        logger.info(
            f"Initializing DepthAwareUNet with {num_classes} classes, "
            f"encoder_depth={encoder_depth}"
        )
        
        # ====================================================================
        # ENCODER - Using ResNet50 backbone
        # ====================================================================
        resnet = models.resnet50(pretrained=use_pretrained_backbone)
        
        # Extract encoder layers
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(
            nn.Sequential(
                resnet.conv1,
                resnet.bn1,
                resnet.relu,
            )
        )  # 3 -> 64
        
        self.encoder_blocks.append(resnet.layer1)  # 64 -> 256
        self.encoder_blocks.append(resnet.layer2)  # 256 -> 512
        self.encoder_blocks.append(resnet.layer3)  # 512 -> 1024
        self.encoder_blocks.append(resnet.layer4)  # 1024 -> 2048
        
        # Freeze backbone if specified
        if freeze_backbone:
            for param in self.encoder_blocks.parameters():
                param.requires_grad = False
            logger.info("Backbone layers frozen")
        
        # Pool to get bottleneck
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Bottleneck processing
        self.bottleneck = ConvBlock(2048, 2048)
        
        # ====================================================================
        # DECODER - Symmetric decoder with skip connections
        # ====================================================================
        encoder_channels = [64, 256, 512, 1024, 2048]
        
        self.decoder_blocks = nn.ModuleList()
        self.skip_blocks = nn.ModuleList()
        
        # Create decoder layers with skip connections
        for i in range(len(decoder_channels)):
            if i == 0:
                in_ch = encoder_channels[-1]
            else:
                in_ch = decoder_channels[i - 1]
            
            skip_ch = encoder_channels[-(i + 2)] if i < len(encoder_channels) - 1 else 0
            
            self.decoder_blocks.append(
                UpBlock(in_ch, decoder_channels[i], skip_channels=skip_ch)
            )
        
        # ====================================================================
        # SEGMENTATION HEAD (Class prediction)
        # ====================================================================
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1),
        )
        
        # ====================================================================
        # DEPTH HEAD (Pseudo-depth estimation)
        # ====================================================================
        self.depth_head = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),  # Output in [0, 1] range
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the model.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            
        Returns:
            Tuple of (logits, depth)
                - logits: Class logits (B, num_classes, H, W)
                - depth: Pseudo-depth map (B, 1, H, W)
        """
        # ====================================================================
        # ENCODER
        # ====================================================================
        encoder_outputs = []
        h = x
        
        for i, block in enumerate(self.encoder_blocks):
            h = block(h)
            encoder_outputs.append(h)
            if i < len(self.encoder_blocks) - 1:
                h = self.pool(h)
        
        # ====================================================================
        # BOTTLENECK
        # ====================================================================
        h = self.bottleneck(h)
        
        # ====================================================================
        # DECODER
        # ====================================================================
        for i, (decoder_block) in enumerate(self.decoder_blocks):
            # Get skip connection from encoder
            skip = encoder_outputs[-(i + 2)] if i < len(encoder_outputs) - 1 else None
            h = decoder_block(h, skip)
        
        # ====================================================================
        # HEADS
        # ====================================================================
        # Segmentation output
        class_logits = self.segmentation_head(h)
        
        # Depth output
        depth = self.depth_head(h)
        
        return class_logits, depth
    
    def get_model_size(self) -> Tuple[int, int]:
        """
        Calculate model size in MB.
        
        Returns:
            Tuple of (total_params, size_mb)
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Approximate size in MB (assuming float32 = 4 bytes per parameter)
        size_mb = (total_params * 4) / (1024 ** 2)
        
        logger.info(
            f"Model size: {size_mb:.2f}MB "
            f"({total_params:,} total params, "
            f"{trainable_params:,} trainable)"
        )
        
        return total_params, trainable_params


def create_model(
    num_classes: int = config.NUM_CLASSES,
    pretrained: bool = config.PRETRAINED,
    freeze_backbone: bool = config.FREEZE_BACKBONE,
) -> DepthAwareUNet:
    """
    Create and return a DepthAwareUNet model.
    
    Args:
        num_classes: Number of segmentation classes
        pretrained: Whether to use pretrained backbone
        freeze_backbone: Whether to freeze backbone layers
        
    Returns:
        Initialized DepthAwareUNet model
    """
    model = DepthAwareUNet(
        num_classes=num_classes,
        use_pretrained_backbone=pretrained,
        freeze_backbone=freeze_backbone,
    )
    
    model.get_model_size()
    
    return model
