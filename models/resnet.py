import torchvision
import torch.nn as nn


class ResNetFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, no_grad=False, device=None):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.classifier = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x