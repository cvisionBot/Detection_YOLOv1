from torch import optim
from models.backbone.yolov1 import YOLOv1_B
from models.head.yolov1_head import YOLOv1

def get_model(model_name):
    model_dict = {'YOLOv1':YOLOv1_B}
    return model_dict.get(model_name)

def get_optimizer(optimizer_name, params, **kwargs):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    optimizer = optim_dict.get(optimizer_name)
    if optimizer:
        return optimizer(params, **kwargs)