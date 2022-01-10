from torch import optim
from models.backbone.baseline import BaseNet
def get_model(model_name):
    model_dict = {'BaseNet', BaseNet}
    return model_dict.get(model_name)

def get_optimizer(optimizer_name):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    return optim_dict.get(optimizer_name)
