import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from models.initialize import weight_initialize
from utils.module_select import get_model


class SexAge(nn.Module):
    def __init__(self, backbone, in_channels, input_size):
        super().__init__()

        self.backbone = backbone(in_channels).features
        c = self.backbone(torch.randn((1, 3, input_size*2, input_size), dtype=torch.float32)).size(1)

        self.sex_head = nn.Sequential(
            nn.Conv2d(c, 1280, 1, 1, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 2, 1)
        )

        self.age_head = nn.Sequential(
            nn.Conv2d(c, 1280, 1, 1, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, 6, 1)
        )
        
        weight_initialize(self.sex_head)
        weight_initialize(self.age_head)

    def forward(self, x):
        # backbone forward
        x = self.backbone(x)

        # prediction
        pred_sex = self.sex_head(x)
        pred_sex = pred_sex.view(pred_sex.size(0), pred_sex.size(1))

        pred_age = self.age_head(x)
        pred_age = pred_age.view(pred_age.size(0), pred_age.size(1))

        return pred_sex, pred_age


if __name__ == '__main__':
    input_size = 224
    in_channels = 3

    backbone = get_model('darknet19')

    model = SexAge(
        backbone=backbone,
        in_channels=in_channels,
        input_size=input_size
    )

    torchsummary.summary(model, (3, input_size*2, input_size), batch_size=1, device='cpu')

    print(model(torch.randn(1, 3, input_size, input_size))[0].size())
    print(model(torch.randn(1, 3, input_size, input_size))[1].size())
