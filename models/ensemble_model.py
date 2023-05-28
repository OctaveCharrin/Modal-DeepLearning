import torch
import torch.nn as nn

class EnsembleModel(nn.Module):
    def __init__(self, model1, model2, model1txt, model2txt, num_classes):
        super(EnsembleModel, self).__init__()

        # Freeze the parameters of the backbone models
        for param in model1.parameters():
            param.requires_grad = False
        for param in model2.parameters():
            param.requires_grad = False

        # Store the backbone models
        self.model1 = model1
        self.model2 = model2
        self.model1txt = model1txt
        self.model2txt = model2txt

        # Learnable fusion layer
        self.fusion_layer = nn.Linear(in_features=num_classes * 2, out_features=num_classes)
        
        # Initialize the fusion layer weights and biases for sum operation
        with torch.no_grad():
            eye_matrix1 = torch.eye(num_classes)
            eye_matrix2 = torch.eye(num_classes)
            stacked_matrix = torch.cat((eye_matrix1, eye_matrix2), dim=1)

            self.fusion_layer.weight.copy_(stacked_matrix)
            self.fusion_layer.bias.copy_(torch.zeros(num_classes))

    def forward(self, x):
        # Forward pass through the backbone models
        output1, _ = self.model1(x, self.model1txt)
        output2, _ = self.model2(x, self.model2txt)

        # Concatenate the outputs along the channel dimension
        concatenated = torch.cat((output1, output2), dim=1)

        # Pass the concatenated output through the fusion layer
        fused_output = self.fusion_layer(concatenated)

        return fused_output