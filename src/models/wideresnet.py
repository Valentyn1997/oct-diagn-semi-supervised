import torch
import torch.nn as nn


class WideResNet_50_2(nn.Module):
    def __init__(self, num_classes, pretrained=False, freeze_layers=False):
        super(WideResNet_50_2, self).__init__()
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        # load WRN-50-2:
        self.wrn = torch.hub.load('pytorch/vision:v0.5.0', 'wide_resnet50_2', pretrained=pretrained)
        self.wrn.eval()
        self.wrn.fc = nn.Linear(self.wrn.fc.in_features, num_classes)

    def forward(self, x):
        if self.freeze_layers:
            for name, param in self.wrn.named_parameters():
                if name == 'fc.weight' or name == 'fc.bias':
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        return self.wrn(x)

