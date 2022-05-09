import sys
import os
sys.path.append(os.getcwd())

from models.layers.conv_block import Conv2dBnRelu
from models.initialize import weight_initialize
from torch import nn
import torchsummary

cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class _VGG(nn.Module):
    def __init__(self, in_channels, num_classes, cfg, batch_norm):
        super(_VGG, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.features = self._make_layers(cfg, batch_norm)

        self.classifier = nn.Sequential(
            Conv2dBnRelu(512, 1280, 1),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(1280, num_classes, 1)
        )

    def forward(self, x):
        x = self.features(x)
        pred = self.classifier(x)
        b, c, _, _ = pred.size()
        pred = pred.view(b, c)

        return {'pred': pred}

    def _make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = self.in_channels
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
                else:
                    layers += [conv2d, nn.ReLU()]
                in_channels = v
        return nn.Sequential(*layers)

def vgg16(in_channels, num_classes=1000):
    model = _VGG(in_channels, num_classes, cfgs['D'], False)
    weight_initialize(model)
    return model

def vgg16_bn(in_channels, num_classes=1000):
    model = _VGG(in_channels, num_classes, cfgs['D'], True)
    weight_initialize(model)
    return model


if __name__ == '__main__':
    model = vgg16_bn(in_channels=3)
    # print(model(torch.rand(1, 3, 64, 64)))
    print(model)
    torchsummary.summary(model, (3, 448, 448), batch_size=1, device='cpu')

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

    from torchvision import models

    model =models.resnet50()
    model = nn.Sequential(*list(model.children())[:-2])
    torchsummary.summary(model, (3, 224, 224), batch_size=1, device='cpu')