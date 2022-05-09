import sys
import os
sys.path.append(os.getcwd())

import torch
from torch import nn
import torchsummary

from models.layers.conv_block import Conv2dBnRelu
from models.initialize import weight_initialize


class _Darknet19(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(_Darknet19, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = nn.Sequential(
            Conv2dBnRelu(in_channels, 32, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(32, 64, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(64, 128, 3),
            Conv2dBnRelu(128, 64, 1),
            Conv2dBnRelu(64, 128, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(128, 256, 3),
            Conv2dBnRelu(256, 128, 1),
            Conv2dBnRelu(128, 256, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(256, 512, 3),
            Conv2dBnRelu(512, 256, 1),
            Conv2dBnRelu(256, 512, 3),
            Conv2dBnRelu(512, 256, 1),
            Conv2dBnRelu(256, 512, 3),
            nn.MaxPool2d(2, 2),
            
            Conv2dBnRelu(512, 1024, 3),
            Conv2dBnRelu(1024, 512, 1),
            Conv2dBnRelu(512, 1024, 3),
            Conv2dBnRelu(1024, 512, 1),
            Conv2dBnRelu(512, 1024, 3)
        )

        self.classifier = nn.Sequential(
            Conv2dBnRelu(1024, num_classes, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def darknet19(in_channels, num_classes=1000):
    model = _Darknet19(in_channels, num_classes)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    input_size = 64
    in_channels = 3
    num_classes = 200
    
    model = darknet19(in_channels, num_classes)
    
    torchsummary.summary(model, (in_channels, input_size, input_size), batch_size=1, device='cpu')
    
    # model = model.features[:17]
    # torchsummary.summary(model, (3, input_size, input_size), batch_size=1, device='cpu')
    
    # print(list(model.children()))
    # print(f'\n-------------------------------------------------------------\n')
    # new_model = nn.Sequential(*list(model.children())[:-1])
    # print(new_model.modules)
    
    # for idx, child in enumerate(model.children()):
    #     print(child)
    #     if idx == 0:
    #         for i, param in enumerate(child.parameters()):
    #             print(i, param)
    #             param.requires_grad = False
    #             if i == 4:
    #                 break

    # torchsummary.summary(model, (3, 64, 64), batch_size=1, device='cpu')
    
    # from torchvision import models

    # model = models.resnet18(num_classes=200)
    # models.vgg16
    # # model = models.resnet50(num_classes=200)
    # model = models.efficientnet_b0(num_classes=200)
    # torchsummary.summary(model, (3, 64, 64), batch_size=1, device='cpu')

