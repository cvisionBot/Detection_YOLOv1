import torch
from torch import nn


# Conv2d Block Module - yolov1 activation = ReLu
class Conv2dBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(Conv2dBnRelu, self).__init__() 
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                                    groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        output = self.conv(x)
        output = self.bn(output)
        output = self.activation(output)
        return output 