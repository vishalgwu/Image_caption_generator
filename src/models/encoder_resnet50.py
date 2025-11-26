# src/models/encoder_resnet50.py
import torch
import torch.nn as nn
from torchvision import models


class ResNet50Encoder(nn.Module):
    def __init__(self, train_cnn: bool = False):
        super().__init__()

        # pretrained ImageNet ResNet-50
        backbone = models.resnet50(pretrained=True)

        # remove final FC, keep features only
        modules = list(backbone.children())[:-1]  # drop fc
        self.backbone = nn.Sequential(*modules)
        self.out_dim = backbone.fc.in_features  # 2048

        if not train_cnn:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        """
        x: (B, 3, H, W)
        returns: (B, 2048)
        """
        feats = self.backbone(x)  # (B, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)
        return feats
