import torch
import torch.nn as nn
import pytorch_lightning as pl

from models.loss.yolov1_loss import Yolov1_Loss

class YOLO_Detector(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.save_hyperparameters(ignore='model')
        self.model = model
        self.loss_fn = Yolov1_Loss(cfg)

    def forward(self, x):
        pred_output = self.model(x)
        return pred_output

    def training_step(self, batch, batch_idx):
        loss = self.opt_training_step(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def opt_training_step(self, batch):
        prediction = self.model(batch['img'])
        loss = self.loss_fn([prediction, batch])
        return loss

    def validation_step(self, batch, batch_idx):
        val_prediction = self.model(batch['img'])
        val_loss = self.loss_fn([val_prediction, batch])
        self.log('val_loss', val_loss, logger=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return val_loss

    def configure_optimizer(self):
        cfg = self.hparam.cfg
        epoch_length = self.hparams.epoch_length
        


    @staticmethod
    def disable_running_stats(model):
        def _disable(module):
            if isinstance(module, nn.BatchNorm2d):
                module.backup_momentum = module.momentum
                module.momentum = 0

        model.apply(_disable)

    @staticmethod
    def enable_running_stats(model):
        def _enable(module):
            if isinstance(module, nn.BatchNorm2d) and hasattr(module, "backup_momentum"):
                module.momentum = module.backup_momentum

        model.apply(_enable)