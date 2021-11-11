import platform
import argparse
import albumentations
import albumentations.pytorch

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from dataset import data
from utils.utility import make_model_name
from utils.module_select import get_model, get_optimizer
from utils.yaml_helper import get_train_configs

from models.head.yolov1_head import YOLOv1
from module.detector import YOLO_Detector

def train(cfg, ckpt=None):
    input_size = cfg['input_size']

    # Test Overfit
    # train_transforms = albumentations.Compose([
    #     albumentations.Resize(input_size, input_size, always_apply=True),
    #     albumentations.Normalize(0, 1),
    #     albumentations.pytorch.ToTensorV2(),
    # ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(input_size, input_size, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    valid_transform = albumentations.Compose([
        albumentations.Resize(input_size, input_size, always_apply=True),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    data_module = data.YoloFormat(
        train_list=cfg['train_list'], val_list=cfg['val_list'],
        workers=cfg['workers'], batch_size=cfg['batch_size'],
        train_transforms=train_transforms, val_transforms=valid_transform
    )

    backbone = get_model(cfg['backbone'])
    yolo = YOLOv1(Backbone=backbone, in_channels=cfg['in_channels'],
                    grid_size=cfg['grid_size'], num_boxes=cfg['num_boxes'], num_classes=cfg['classes'])
    yolo_module = YOLO_Detector(yolo, cfg)

    callbacks = [
        ModelCheckpoint(monitor='val_loss', save_last=True,
                                        every_n_epochs=cfg['save_freq'])
    ]

    trainer = pl.Trainer(
        max_epochs=cfg['epochs'],
        logger=TensorBoardLogger(cfg['save_dir'],
                                 make_model_name(cfg)),
        gpus=cfg['gpus'],
        accelerator='ddp' if platform.system() != 'Windows' else None,
        plugins=DDPPlugin() if platform.system() != 'Windows' else None,
        callbacks=callbacks,
        resume_from_checkpoint=ckpt,
        **cfg['trainer_options'])
    trainer.fit(yolo_module, data_module)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=str,
                        help='Train config file')
    parser.add_argument('--ckpt', required=False, type=str,
                        help='Train checkpoint')

    args = parser.parse_args()
    cfg = get_train_configs(args.cfg)

    train(cfg, args.ckpt)