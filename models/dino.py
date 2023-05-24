import torchvision
import torch.nn as nn
import torch

vitb16 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')


class MeanTeacherModel(nn.Module):
    def __init__(self, num_classes, frozen=False, no_grad=False):
        super().__init__()
        self.backbone = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()
        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(nn.BatchNorm1d(2048),
                                        nn.Dropout(p=0.25),
                                        nn.Linear(in_features=2048, out_features=1024),
                                        nn.ReLU(),
                                        nn.BatchNorm1d(1024, eps=1e-05, momentum=0.1),
                                        nn.Dropout(p=0.5),
                                        nn.Linear(in_features=1024, out_features=num_classes),
                                        )
        if no_grad:
            for param in self.classifier.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x

