from torch import optim
from models.backbone.resnet import ResNet

def get_model(model_name):
    model_dict = {'ResNet': ResNet}
    return model_dict.get(model_name)

def get_optimizer(optimizer_name):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    return optim_dict.get(optimizer_name)
