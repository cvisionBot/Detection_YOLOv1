import math
import torch
from torch._C import DeviceObjType
import torch.nn as nn
import numpy as np



class Yolov1_Loss(nn.Module):
    def __init__(self, cfg=None):
        super(Yolov1_Loss, self).__init__()
        self.S = 7   # cfg['grid_size']
        self.B = 2   # cfg['num_boxes']
        self.C = 20  # cfg['num_classes']
        self.coord = 5
        self.nobj_coord = 0.5
        self.mse_loss = nn.MSELoss(reduction='sum')
        self.final_act = nn.Sigmoid()

    def forward(self, pred, samples):
        # b, 30, 7, 7
        # class(20), objectness0, box0(4), objectness1, box1(4) -> 30, box : cx, cy, w, h
        # box : x, y, w, h
        # annot : x1, y1, w, h, cls  ( image, bboxes )
        predictions = self.final_act(pred)  # .reshape(-1, self.S, self.S, self.C + self.B * 5) 
        cls_pred = predictions[:,:self.C, :, :]
        reg_pred = predictions[:,self.C:, :, :]
        imgs = samples['img']
        annots = samples['annot']
        # device = imgs.device
        losses = torch.zeros(1)
        for b in range(predictions.shape[0]):
            print("Batch {}".format(b))
            gt_bboxes = annots[b]
            gt_bboxes = gt_bboxes[gt_bboxes[:, 4] != -1]
            gt_bboxes = gt_bboxes[:, :4] / 448.
            losses += self.r_loss(predictions[b], imgs[b], annots[b])
            print(losses)
    
    def r_loss(self, pred, img, annots):
        bboxes = annots
        bboxes = bboxes[bboxes[:, 4] != -1]
        cls_target = torch.zeros(self.C, 7, 7, dtype=float)
        positive_idx = torch.zeros(2)
        cls_pred = pred[:self.C, :, :]  # 20, 7, 7
        reg_pred = pred[self.C:, :, :]  # 10, 7, 7
        gt_boxes = torch.zeros(7, 7)
        losses = 0
        for idx, bbox in enumerate(bboxes):
            cls_target[int(bbox[-1]), : , :] = 1.
            positive_idx[0] = int((bbox[0]+(bbox[2]*0.5)) * self.S / 448.) # x idx
            positive_idx[1] = int((bbox[1]+(bbox[3]*0.5)) * self.S / 448.) # y idx
            gt_idx = torch.zeros(7, 7, dtype=float)
            gt_idx[int(positive_idx[0]), int(positive_idx[1])] = 1.
            gt_boxes[gt_idx[:, :]>0] = 1.
            iou = self.calc_iou(reg_pred, img, bbox[:4]/448.)  # 2, 7, 7
            _, best_idx = torch.max(iou, dim = 0)
            # iou, best_iou_box, best_idx, cls_target
            xy_loss0            = self.coord * gt_boxes * best_idx * (self.mse_loss(bbox[0] / 448., reg_pred[6,:,:]) + self.mse_loss(bbox[ 1] / 448., reg_pred[7, : , :]))
            xy_loss1            = self.coord * gt_boxes * (1. - best_idx) * (self.mse_loss(bbox[0]/448., reg_pred[1,:,:]) + self.mse_loss(bbox[ 1] / 448., reg_pred[2, : , :]))
            wh_loss0            = self.coord * gt_boxes * best_idx * (self.mse_loss(torch.sqrt(bbox[ 2] /448.), torch.sqrt(reg_pred[8,:,:])) 
                                                            + self.mse_loss(torch.sqrt(bbox[ 3] /448.), torch.sqrt(reg_pred[9,:,:])))
            wh_loss1            = self.coord * gt_boxes * (1. - best_idx) * (self.mse_loss(torch.sqrt(bbox[ 2] /448.), torch.sqrt(reg_pred[3,:,:])) 
                                                                    + self.mse_loss(torch.sqrt(bbox[ 3] /448.), torch.sqrt(reg_pred[4,:,:])))
            conf_loss0          = gt_boxes * iou[0, : , :] * best_idx * self.mse_loss(torch.tensor([1.]), reg_pred[0, :, :])
            conf_loss1          = gt_boxes * iou[1, : , :] * (1. - best_idx) * self.mse_loss(torch.tensor([1.]), reg_pred[5, :, :])
            no_obj_conf_loss0   = self.nobj_coord * (1. - gt_boxes) * iou[0, : , :] * best_idx * self.mse_loss(torch.tensor([0.]), reg_pred[0, :, :])
            no_obj_conf_loss1   = self.nobj_coord * (1. - gt_boxes) * iou[1, : , :] * (1. - best_idx) * self.mse_loss(torch.tensor([0.]), reg_pred[5, :, :])
            cls_loss            = gt_boxes * self.mse_loss(cls_target ,cls_pred[:20, :, :])
            losses += torch.sum(torch.flatten(xy_loss0+xy_loss1+wh_loss0+wh_loss1+conf_loss0 +conf_loss1+no_obj_conf_loss0+no_obj_conf_loss1+cls_loss))
        
        return losses


    def calc_iou(self, pred_bbox, img, annots): 
        # input : pred[self.C:, :, :], img, annots (x1, y1, w, h, c)
        # output : iou (2, 7, 7) float
                
        x = torch.arange(7)
        y = torch.arange(7)
        grid_y, grid_x = torch.meshgrid(x, y)
        # sig = torch.nn.Sigmoid()
        pred_box0 = pred_bbox[1:5, :, :]
        pred_box1 = pred_bbox[6:10, :, :]
        annots_box_area = (annots[2] * annots[3])
        pred_box0_area = (pred_box0[2, :, :] * pred_box0[3, :, :]) # 10, 7, 7
        pred_box1_area = (pred_box1[2, :, :] * pred_box1[3, :, :]) # 10, 7, 7
        inter_box0_x1 = torch.max(annots[0], pred_box0[0, :, :] - (pred_box0[2, :, :] * 0.5))
        inter_box0_y1 = torch.max(annots[1], pred_box0[1, :, :] - (pred_box0[3, :, :] * 0.5))
        inter_box0_w = torch.min(annots[0]+annots[2], pred_box0[0, :, :] + (pred_box0[2, :, :]*0.5)) - inter_box0_x1
        inter_box0_h = torch.min(annots[1]+annots[3], pred_box0[1, :, :] + (pred_box0[3, :, :]*0.5)) - inter_box0_y1
        inter_box0_area = inter_box0_w.clamp(0) * inter_box0_h.clamp(0)
        
        inter_box1_x1 = torch.max(annots[0], pred_box1[0, :, :] - (pred_box1[2, :, :] * 0.5))
        inter_box1_y1 = torch.max(annots[1], pred_box1[1, :, :] - (pred_box1[3, :, :] * 0.5))
        inter_box1_w = torch.min(annots[0]+annots[2], pred_box1[0, :, :] + (pred_box1[2, :, :]*0.5)) - inter_box1_x1
        inter_box1_h = torch.min(annots[1]+annots[3], pred_box1[1, :, :] + (pred_box1[3, :, :]*0.5)) - inter_box1_y1
        inter_box1_area = inter_box1_w.clamp(0) * inter_box1_h.clamp(0)

        pred_box0_iou = inter_box0_area/(pred_box0_area+annots_box_area-inter_box0_area + 1e-6)
        pred_box1_iou = inter_box1_area/(pred_box1_area+annots_box_area-inter_box1_area + 1e-6)
        pred_box_iou = torch.stack([pred_box0_iou, pred_box1_iou], dim=0)
        return pred_box_iou


if __name__=="__main__":
    import albumentations
    import albumentations.pytorch
    from dataset.utils import collater
    from models.backbone.yolov1 import YOLOv1
    
    from torch.utils.data import Dataset, DataLoader
    from dataset.data import PreDataset
    from dataset.utils import collater
    
    train_transforms = albumentations.Compose([
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    loader = DataLoader(PreDataset(
        transforms=train_transforms, files_list='../dataset/extract_coco_train_test10.txt'),
        batch_size=1, shuffle=True, collate_fn=collater)
        
    model = YOLOv1(in_channels=3)
    pred = model(torch.rand(1, 3, 448, 448))
    # pred = torch.rand(2, 30, 7, 7)
    v1_loss = Yolov1_Loss()
    for batch, sample in enumerate(loader):
        v1_loss(pred, sample)