import torchvision
import torch.nn as nn
import torch
import timm


class ViTFinetune(nn.Module):
    def __init__(self, num_classes, frozen=False, no_grad=False, device=None):
        super().__init__()
        self.backbone = timm.models.vit_base_patch16_224_in21k(pretrained=True)
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.Linear(in_features=21843, out_features=num_classes))

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x