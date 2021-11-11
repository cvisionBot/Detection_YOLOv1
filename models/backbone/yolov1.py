import torch
from torch import nn

from ..layers.conv_block import Conv2dBnRelu
from ..initialize import weight_initialize


class Stem_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Stem_block, self).__init__()
        self.conv = Conv2dBnRelu(in_channels, out_channels, kernel_size, stride, padding)
        self.max_p = nn.MaxPool2d((2, 2), stride=(2, 2))
    
    def forward(self, x):
        output = self.conv(x)
        output = self.max_p(output)
        return output

class Yolov1_block1(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Yolov1_block1, self).__init__()
        self.conv = Conv2dBnRelu(in_channels, out_channels, kernel_size, stride, padding)
        self.max_p = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        output = self.conv(x)
        output = self.max_p(output)
        return output

class Yolov1_block2(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Yolov1_block2, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, 128, (1, 1), stride, 0)
        self.conv2 = Conv2dBnRelu(128, 256, kernel_size, stride, padding)
        self.conv3 = Conv2dBnRelu(256, 256, (1, 1), stride, 0)
        self.conv4 = Conv2dBnRelu(256, out_channels, kernel_size, stride, padding)
        self.max_p = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.max_p(output)
        return output

class Yolov1_block3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Yolov1_block3, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, 256, (1, 1), stride, 0)
        self.conv2 = Conv2dBnRelu(256, 512, kernel_size, stride, padding)
        self.conv3 = Conv2dBnRelu(512, 256, (1, 1), stride, 0)
        self.conv4 = Conv2dBnRelu(256, 512, kernel_size, stride, padding)
        self.conv5 = Conv2dBnRelu(512, 256, (1, 1), stride, 0)
        self.conv6 = Conv2dBnRelu(256, 512, kernel_size, stride, padding)
        self.conv7 = Conv2dBnRelu(512, 256, (1, 1), stride, 0)
        self.conv8 = Conv2dBnRelu(256, 512, kernel_size, stride, padding)
        self.conv9 = Conv2dBnRelu(512, 512, (1, 1), stride, 0)
        self.conv10 = Conv2dBnRelu(512, out_channels, kernel_size, stride, padding)
        self.max_p = nn.MaxPool2d((2, 2), stride=(2, 2))

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        output = self.conv8(output)
        output = self.conv9(output)
        output = self.conv10(output)
        output = self.max_p(output)
        return output

class Yolov1_block4(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Yolov1_block4, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, 512, (1, 1), stride, 0)
        self.conv2 = Conv2dBnRelu(512, 1024, kernel_size, stride, padding)
        self.conv3 = Conv2dBnRelu(1024, 512, (1, 1), stride, 0)
        self.conv4 = Conv2dBnRelu(512, 1024, kernel_size, stride, padding)
        self.conv5 = Conv2dBnRelu(1024, 1024, kernel_size, stride, padding)
        self.conv6 = Conv2dBnRelu(1024, out_channels, kernel_size, 2, padding)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        return output

class Yolov1_block5(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(Yolov1_block5, self).__init__()
        self.conv1 = Conv2dBnRelu(in_channels, 1024, kernel_size, stride, padding)
        self.conv2 = Conv2dBnRelu(1024, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2(output)
        return output

class _YOLOv1_Backbone(nn.Module):
    def __init__(self, in_channels):
        super(_YOLOv1_Backbone, self).__init__()
        self.stem = Stem_block(in_channels, 192, (7, 7), 2, 3)
        self.block1 = Yolov1_block1(192, 256, (3, 3), 1, 1)
        self.block2 = Yolov1_block2(256, 512, (3, 3), 1, 1)
        self.block3 = Yolov1_block3(512, 1024, (3, 3), 1, 1)
        self.block4 = Yolov1_block4(1024, 1024, (3, 3), 1, 1)
        self.block5 = Yolov1_block5(1024, 1024, (3, 3), 1, 1)

    def forward(self, input):
        output = self.stem(input)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        return output

def YOLOv1_B(in_channels):
    model = _YOLOv1_Backbone(in_channels)
    weight_initialize(model)
    return model

if __name__ == '__main__':
    model = YOLOv1_B(in_channels=3)
    print(model(torch.rand(1, 3, 224, 224)))
