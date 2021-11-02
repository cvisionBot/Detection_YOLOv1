import math
import torch
import torch.nn as nn
from anchors import Anchors

class Yolov1_Loss(nn.Module):
    def __init__(self, cfg):
        super(Yolov1_Loss, self).__init__()
        self.anchors = Anchors(cfg['grid_size'], cfg['num_box'], cfg['num_class'])    
        self.obj_coord = 0.5
        self.nobj_coord = 0.5
        # Anchor 매칭

        # Config[grid_size]
