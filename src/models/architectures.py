"""Model architectures for DeepShield AI baseline and forensic classifiers.

Implements MesoInception-4 (Afchar et al.) and EfficientNet-B0 binary classifier.
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models


class MesoInceptionBlock(nn.Module):
    """Inception module used in MesoInception-4."""
    def __init__(self, in_channels: int, a1: int, b1: int, b2: int, c1: int, c2: int, d1: int, d2: int):
        super().__init__()
        # 1x1 conv
        self.conv1 = nn.Conv2d(in_channels, a1, kernel_size=1, padding=0)
        # 1x1 conv -> 3x3 conv
        self.conv2_1 = nn.Conv2d(in_channels, b1, kernel_size=1, padding=0)
        self.conv2_2 = nn.Conv2d(b1, b2, kernel_size=3, padding=1)
        # 1x1 conv -> 5x5 conv
        self.conv3_1 = nn.Conv2d(in_channels, c1, kernel_size=1, padding=0)
        self.conv3_2 = nn.Conv2d(c1, c2, kernel_size=5, padding=2)
        # 3x3 maxpool -> 1x1 conv
        self.pool4 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels, d1, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(x))))
        x3 = self.relu(self.conv3_2(self.relu(self.conv3_1(x))))
        x4 = self.relu(self.conv4(self.pool4(x)))
        return torch.cat([x1, x2, x3, x4], dim=1)


class MesoInception4(nn.Module):
    """MesoInception-4 deepfake detection architecture (Afchar et al., 2018).
    
    Designed to detect micro-texture manipulation at low computational cost.
    Input: [B, 3, 256, 256], Output: [B, 2] (Logits for Real, Fake)
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.inc1 = MesoInceptionBlock(3, 1, 1, 2, 1, 2, 1, 2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm1 = nn.BatchNorm2d(11)

        self.inc2 = MesoInceptionBlock(11, 2, 2, 4, 1, 4, 1, 4)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm2 = nn.BatchNorm2d(14)

        self.conv3 = nn.Conv2d(14, 16, kernel_size=5, padding=2)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.norm3 = nn.BatchNorm2d(16)

        self.conv4 = nn.Conv2d(16, 16, kernel_size=5, padding=2)
        self.pool4 = nn.MaxPool2d(kernel_size=4, stride=4)
        self.norm4 = nn.BatchNorm2d(16)

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(16 * 8 * 8, 16)
        self.leaky_relu = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(self.pool1(self.inc1(x)))
        x = self.norm2(self.pool2(self.inc2(x)))
        x = self.norm3(self.pool3(self.relu(self.conv3(x))))
        x = self.norm4(self.pool4(self.relu(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class EfficientNetB0Deepfake(nn.Module):
    """EfficientNet-B0 adapted for binary facial manipulation forensics."""
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False):
        super().__init__()
        weights = tv_models.EfficientNet_B0_Weights.DEFAULT if pretrained_backbone else None
        self.backbone = tv_models.efficientnet_b0(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)
