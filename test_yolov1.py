import os
import cv2
import torch
import random
import argparse
import numpy as np

from models.head.yolov1_head import YOLOv1
from module.detector import YOLO_Detector
from utils.module_select import get_model
from utils.utility import preprocess_input
from utils.yaml_helper import get_train_configs

def parse_names(names_file):
    names_file = os.getcwd() + names_file
    with open(names_file, 'r') as f:
        return f.read().splitlines()

def gen_random_colors(names):
    colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for i in range(len(names))]
    return colors

def visualize_detection(image, box, class_name, conf, color):
    x1, y1, x2, y2 = box
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color)

    caption = f'{class_name} {conf:.2f}'
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
    image = cv2.putText(image, caption, (x1+4, y1+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return image

def main(cfg, image_name, save):
    names = parse_names(cfg['names'])
    colors = gen_random_colors(names)

    # Preprocess Image
    image = cv2.imread(image_name)
    image = cv2.resize(image, (448, 448))
    image_inp = preprocess_input(image)
    image_inp = image_inp.unsqueeze(0)
    if torch.cuda.is_available:
        image_inp = image_inp.cuda()

    # Load trained Model
    backbone = get_model(cfg['backbone'])
    yolo = YOLOv1(Backbone=backbone, in_channels=cfg['in_channels'],
                    grid_size=cfg['grid_size'], num_boxes=cfg['num_boxes'], num_classes=cfg['classes'])
    if torch.cuda.is_available:
        yolo = yolo.to('cuda')

    yolo_module = YOLO_Detector.load_from_checkpoint(
        '/home/PYTORCH_YOLOV1/saved/YOLO_PASCAL/version_0/checkpoints/last.ckpt',
        model=yolo
    )

    prediction = yolo_module(image_inp)
    print(prediction)
    prediction = yolo_postprocessing(prediction)

    for p in prediction:
        x1, y1, width, height, conf, class_ = p
        box = [x1, y1, x1 + width, y1 + height]
        name = names[int(class_)]
        color = colors[int(class_)]
        image = visualize_detection(image, box, name, conf, color)
    
    if save:
        cv2.imwrite('./saved/inference.png', image)

def encode_box(pred):
    x1 = pred[0, :, :]
    y1 = pred[1, :, :]
    width = pred[2, :, :]
    height = pred[3, :, :]
    conf = pred[4, :, :]
    encode_box = torch.Tensor([[x1, y1, width, height, conf]])
    return encode_box

def yolo_postprocessing(prediction, grid_size, confidence=0.7):
    '''
        7 x 7 x 30 = model output tensor
    '''
    pred_tensor = []
    for i in range(grid_size):
        for j in range(grid_size):
            if prediction[5, i, j] >= confidence or prediction[10, i, j] >= confidence:
                pred_tensor.append(prediction[:, i, j])
    pred_tensor = torch.stack(pred_tensor)

    encodes = []
    for pred in pred_tensor:
        if pred[5, :, :] >= pred[10, :, :]:
            pred_box = encode_box(pred[0:5, :, :])
        else:
            pred_box = encode_box(pred[5:10, :, :])
        pred_cls = pred.argmax(pred[10:20, :, :])
        encode = torch.cat((pred_box, pred_cls), axis=0)
        encodes.append(encode)
    encodes = torch.stack(encodes)
    return encodes