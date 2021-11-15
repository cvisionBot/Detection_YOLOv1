import torch
import torch.nn as nn

from models.initialize import weight_initialize


class Yolov1_Head(nn.Module):
    def __init__(self, grid_size, num_boxes, num_classes):
        super(Yolov1_Head, self).__init__()
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes
        self.flatten1 = nn.Linear(1024 * self.S * self.S, 4096)
        self.flatten2 = nn.Linear(4096, self.S * self.S * (self.B * 5  + self.C))
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):
        output = self.flatten1(x)
        output = self.leaky_relu(output)
        output = self.flatten2(output)
        b, _, _, _ = output.shape
        output = output.contiguous().view(b, (self.B * 5 + self.C), 7, 7)
        return output

class YOLOv1(nn.Module):
    def __init__(self, Backbone, grid_size=7, num_boxes=2, num_classes=20, in_channels=3):
        super(YOLOv1, self).__init__()
        self.backbone = Backbone(in_channels)
        self.head = Yolov1_Head(grid_size, num_boxes, num_classes)

    def forward(self, input):
        output = self.backbone.stem(input)
        output = self.backbone.block1(output)
        output = self.backbone.block2(output)
        output = self.backbone.block3(output)
        output = self.backbone.block4(output)
        output = self.backbone.block5(output)
        output = torch.flatten(output, 1)
        output = self.head(output)

        return output

if __name__ == '__main__':
    from models.backbone.yolov1 import _YOLOv1_Backbone
    model = YOLOv1(
        Backbone = _YOLOv1_Backbone
    )
    print(model(torch.rand(1, 3, 448, 448)))