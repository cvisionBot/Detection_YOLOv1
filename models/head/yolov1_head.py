import torch
import torch.nn as nn

from models.initialize import weight_initialize


class Yolov1_Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Yolov1_Head, self).__init__()
        self.flatten1 = nn.Linear(in_channels, 4096)
        self.flatten2 = nn.Linear(4096, out_channels)

    def forward(self, x):
        output = self.flatten1(x)
        output = self.flatten2(output)
        # return output.contiguous().view(b, 30, 7, 7)
        output = output.contiguous().view(30, 7, 7)
        return output

class YOLOv1(nn.Module):
    def __init__(self, Backbone, grid_size, num_box, num_classes, in_channels=3):
        super(YOLOv1, self).__init__()
        self.S = grid_size
        self.B = num_box
        self.C = num_classes
        self.backbone = Backbone(in_channels)
        self.head = Yolov1_Head(1024 * self.S * self.S, self.S * self.S * (self.B * 5 + self.C))

    def forward(self, input):
        output = self.backbone.stem(input)
        print('Stem Block Output : ',output.shape)
        output = self.backbone.block1(output)
        print('Block1 Output : ', output.shape)
        output = self.backbone.block2(output)
        print('Block2 Output : ', output.shape)
        output = self.backbone.block3(output)
        print('Block3 Output : ', output.shape)
        output = self.backbone.block4(output)
        print('Block4 Output : ', output.shape)
        output = self.backbone.block5(output)
        print('Block5 Output : ', output.shape)
        output = torch.flatten(output, 1)
        print('Flatten Output : ', output.shape)
        output = self.head(output)
        print('Head Output : ', output.shape)

        return output

if __name__ == '__main__':
    from models.backbone.yolov1 import _YOLOv1_Backbone
    model = YOLOv1(
        Backbone = _YOLOv1_Backbone,
        grid_size = 7,
        num_box = 2,
        num_classes = 20,
        in_channels = 3
    )
    print(model(torch.rand(1, 3, 448, 448)))