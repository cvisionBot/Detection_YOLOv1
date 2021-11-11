import torch
import numpy as np

def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v =max(min_value, int(v+divisor/2)//divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

# v 값을 divisor 배수로 나눠서지게 해서 가는 거
# v 14면 16으로 올려서 나간다.
# 채널 수 계산할 때 들어간다 width multiply에 대해 채널에 상수 곱으로 진행 그래서 8의 배수로 올림
# 8의 배수로 되어야지 처리하는게 빨라진다 (메모리 단위가 그렇기 때문에)
# channel + cosine anearling

def make_model_name(cfg):
    return cfg['model'] + '_' + cfg['dataset_name']


def preprocess_input(image, mean=0., std=1., max_pixel=255.):
    normalized = (image.astype(np.float32) - mean *
                  max_pixel) / (std * max_pixel)
    return torch.tensor(normalized).permute(2, 0, 1)
