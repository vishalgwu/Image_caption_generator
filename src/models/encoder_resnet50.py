# src/models/encoder_resnet50.py
import torch.nn as nn
import torchvision.models as models


class ResNet50Encoder(nn.Module):
    def __init__(self, train_cnn: bool = False):
        """
        ResNet-50 encoder that outputs a 2048-dim feature vector per image.
        If train_cnn=False, all ResNet weights are frozen.
        """
        super().__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

        # Remove the final fully-connected classification layer
        modules = list(resnet.children())[:-1]   # everything except the last FC
        self.cnn = nn.Sequential(*modules)

        # Optionally freeze CNN
        if not train_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

    def forward(self, images):
        """
        images: (B, 3, 224, 224)
        returns: (B, 2048)
        """
        feats = self.cnn(images)          # (B, 2048, 1, 1)
        feats = feats.view(feats.size(0), -1)
        return feats
