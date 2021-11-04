import math
import torch
import torch.nn as nn


class Yolov1_Loss(nn.Module):
    def __init__(self, cfg):
        super(Yolov1_Loss, self).__init__()
        self.S = cfg['grid_size']
        self.B = cfg['num_boxes']
        self.C = cfg['num_classes']

        self.obj_coord = 0.5
        self.nobj_coord = 0.5
        self.MSELoss(reduction='sum')

    def forward(self, pred, target):
        predictions = pred.reshape(-1, self.S, self.S, self.C + self.B * 5)
        
        loss = ''
        return loss