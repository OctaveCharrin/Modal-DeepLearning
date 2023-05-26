import torchvision
import torch.nn as nn
import torch


class DinoModel(nn.Module):
    def __init__(self, num_classes, frozen=False, no_grad=False):
        super().__init__()
        self.backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(in_features=768, out_features=num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x