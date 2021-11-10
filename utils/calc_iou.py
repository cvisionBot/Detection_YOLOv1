import math
import torch
import torch.nn as nn

def calc_box_iou(pred_box, gt_box):
    x1 = torch.maximum(pred_box[0], gt_box[0])
    y1 = torch.maximum(pred_box[1], gt_box[1])
    x2 = torch.minimum(pred_box[2], gt_box[2])
    y2 = torch.minimum(pred_box[3], gt_box[3])

    intersection = torch.maximum(x2 - x1, torch.zeros_like(x1) * torch.maximum(y2 - y1, torch.zeros_like(y1)))
    pred_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    union = pred_area + gt_area - intersection

    iou = intersection / union
    return iou
