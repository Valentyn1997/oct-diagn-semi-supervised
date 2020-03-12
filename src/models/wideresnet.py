import torch
import torch.nn as nn


class WideResNet_50_2(nn.Module):
    def __init__(self, num_classes, pretrained=False):
        super(WideResNet_50_2, self).__init__()
        self.pretrained = pretrained
        # load WRN-50-2:
        self.wrn = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
        self.wrn.eval()

        if self.pretrained:
            for param in self.wrn.parameters():
                param.requires_grad = False

        self.wrn.fc = nn.Linear(self.wrn.fc.in_features, num_classes)

    def forward(self, x):
        return self.wrn(x)

