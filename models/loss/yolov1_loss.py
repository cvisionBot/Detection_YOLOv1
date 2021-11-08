import math
import torch
import torch.nn as nn
import numpy as np

from utils.calc_iou import calc_box_iou

class Yolov1_Loss(nn.Module):
    def __init__(self, cfg=None):
        super(Yolov1_Loss, self).__init__()
        self.S = 7 #cfg['grid_size']
        self.B = 2 #cfg['num_boxes']
        self.C = 20 #cfg['num_classses']

        self.obj_coord = 5.0
        self.nobj_coord = 0.5
        self.Sigmoid = nn.Sigmoid()
        self.MSELoss = nn.MSELoss()

    def forward(self, pred, gt_input):
        batch_size = pred.shape[0]
        imgs = gt_input['img']
        annots = gt_input['annot']
        # device = imgs.device
        total_loss = []

        for b in range(batch_size):
            prediction = pred[b]
            prediction = self.Sigmoid(prediction)
            '''
                prediction = [batch : 30 : 7 : 7]
                boxes1 = [batch : 5 : 7 : 7]
                boxes2 = [batch : 10 : 7 : 7]
                class = [batch : 20 : 7 : 7]
            '''

            gt_boxes = annots[b]
            gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]
            target_idx = self.get_target_idx(gt_boxes)

            pred_boxes_ = self.encode_box(prediction, target_idx)
            pred_boxes = []
            for i, gt_box in enumerate(gt_boxes):
                pred_box1_iou = calc_box_iou(pred_boxes_[0], gt_box)
                pred_box2_iou = calc_box_iou(pred_boxes_[1], gt_box)
                box =pred_boxes_[0] if pred_box1_iou > pred_box2_iou else pred_boxes_[1]
                pred_boxes.append(box)
            pred_boxes = torch.stack(pred_boxes)
            print('# # # # # pred_boxes : ', pred_boxes)
            print('# # # # # gt_boxes : ', gt_boxes)
            
            for gt_box, pred_box in zip(gt_boxes, pred_boxes):
                xy_loss = self.obj_coord * (self.MSELoss(gt_box[0], pred_box[0]) 
                                                + self.MSELoss(gt_box[1], pred_box[1]))
                print('# # # loss_xy : ', xy_loss)
                wh_loss = self.obj_coord * (self.MSELoss(torch.sqrt(gt_box[2]), torch.sqrt(pred_box[2]))
                                                + self.MSELoss(torch.sqrt(gt_box[3]), torch.sqrt(pred_box[3])))
                print('# # #  loss_wh : ', wh_loss)
                obj_loss = self.MSELoss(gt_box[4], pred_box[4])
                print('# # # obj_loss : ', obj_loss)
                gt_box[4] = 1.
                nobj_loss = self.nobj_coord * self.MSELoss(gt_box[4], pred_box[4])
                print('# # # noobj loss : ', nobj_loss)
                class_loss = 0
                loss = (xy_loss + wh_loss + obj_loss + nobj_loss + class_loss)
                print('# # # one batch loss : ', loss)
                total_loss.append(loss)

        total_loss = torch.stack(total_loss)
        print('total_loss before sum : ', total_loss)
        total_loss = total_loss.sum()
        print('final loss : ', total_loss)
        return total_loss

    def make_box(self, box):
        x1 = box[0] * 448
        y1 = box[1] * 448
        width = box[2] * 448
        height = box[3] * 448
        cls_id = box[4] 
        decode_box = torch.tensor([[x1, y1, width, height, cls_id]])
        return decode_box

    def encode_box(self, prediction, target_idx):
        for i, p_idx in enumerate(target_idx):
            num_box1 = prediction[0:5, p_idx[0], p_idx[1]]
            num_box2 = prediction[5:10, p_idx[0], p_idx[1]]
            encode_box1 = self.make_box(num_box1)
            encode_box2 = self.make_box(num_box2)
            pred_boxes = torch.cat((encode_box1, encode_box2), axis=0)
        return pred_boxes

    def get_target_idx(self, bboxes):
        bboxes = bboxes[:, :4] / 448.
        gt_idx = torch.zeros(len(bboxes), 2, dtype=int)

        for idx, bbox in enumerate(bboxes):
            gt_idx[idx, 0] = int((bbox[0]+(bbox[2]*0.5)) * self.S) # x idx
            gt_idx[idx, 1] = int((bbox[1]+(bbox[3]*0.5)) * self.S) # y idx
        return gt_idx


if __name__=="__main__": 
    import albumentations 
    import albumentations.pytorch 
    from dataset.utils import collater 
    from models.backbone.yolov1 import YOLOv1 
    from torch.utils.data import Dataset, DataLoader
    from dataset.data import PreDataset
     
    train_transforms = albumentations.Compose([ 
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1)) 
 
    loader = DataLoader(PreDataset( 
        transforms=train_transforms, files_list='/mnt/extract_coco_train_test10.txt'), 
        batch_size=1, shuffle=True, collate_fn=collater) 
         
    model = YOLOv1(in_channels=3) 
    pred = model(torch.rand(1, 3, 448, 448)) 
    # pred = torch.rand(2, 30, 7, 7) 
    v1_loss = Yolov1_Loss() 
    for batch, sample in enumerate(loader): 
        v1_loss(pred, sample)