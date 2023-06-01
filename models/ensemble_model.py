import torch
import torch.nn as nn


class EnsembleModelList(nn.Module):
    def __init__(self, models, num_classes, device=None):
        super(EnsembleModelList, self).__init__()

        assert len(models) != 0

        self.num_models = len(models)

        for model in models:
            for param in model.parameters():
                param.requires_grad = False

        self.models = models

        self.fusion_layer = nn.Linear(in_features=num_classes * len(models), out_features=num_classes)
        
        # Initialize the fusion layer weights and biases for simple sum operation
        with torch.no_grad():
            stacked_matrix=torch.eye(num_classes)
            for _ in range(len(models)-1):
                eye_matrix = torch.eye(num_classes)
                stacked_matrix = torch.cat((stacked_matrix, eye_matrix), dim=1)

            self.fusion_layer.weight.copy_(stacked_matrix)
            self.fusion_layer.bias.copy_(torch.zeros(num_classes))

    def forward(self, x):

        concatenated = self.models[0](x)
        for i in range(1, self.num_models):
            output = self.models[i](x)
            concatenated = torch.cat((concatenated, output), dim=1)

        fused_output = self.fusion_layer(concatenated)

        return fused_output