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
    
            gt_boxes = annots[b]
            gt_boxes = gt_boxes[gt_boxes[:, 4] != -1]
            target_idx = self.get_target_idx(gt_boxes)
            pred_boxes_ = self.encode_box(prediction, target_idx)

            pred_boxes = []
            for i, gt_box in enumerate(gt_boxes):
                pred_box1_iou = calc_box_iou(pred_boxes_[0], gt_box)
                pred_box2_iou = calc_box_iou(pred_boxes_[1], gt_box)
                info =pred_boxes_[0] if pred_box1_iou > pred_box2_iou else pred_boxes_[1]
                pred_boxes.append(info)
            pred_boxes = torch.stack(pred_boxes)
            pred_cls = self.encode_cls(prediction, target_idx)
            pred = torch.cat((pred_boxes, pred_cls), axis=1)

            non_pred_ = self.nencode_box(prediction, target_idx, self.S)
            non_cls = self.nencode_cls(prediction, target_idx, self.S)
            npred = torch.cat((non_pred_, non_cls), axis=1)
   
            for gt_box, pred_box in zip(gt_boxes, pred):
                xy_loss = self.obj_coord * (self.MSELoss(gt_box[0]/448 , pred_box[0] ) 
                                                + self.MSELoss(gt_box[1]/448 , pred_box[1]))
                print('# # # xy_loss : ', xy_loss)
                wh_loss = self.obj_coord * (self.MSELoss(torch.sqrt(gt_box[2]/448), torch.sqrt(pred_box[2]))
                                                + self.MSELoss(torch.sqrt(gt_box[3]/448), torch.sqrt(pred_box[3])))
                print('# # # wh_loss : ', wh_loss)
                obj_loss = self.MSELoss(torch.ones_like(pred_box[4]), pred_box[4])
                print('# # # obj_loss : ', obj_loss)

                nobj_loss = 0
                for npred_box in npred:
                    nobj_loss = self.nobj_coord * self.MSELoss(torch.zeros_like(npred_box[4]), npred_box[4])
                    nobj_loss += nobj_loss
                print('# # # noobj loss : ', nobj_loss)

                class_loss = 0
                for i in range(self.C):
                    if i == gt_box[4]:
                        tclass_loss = self.MSELoss(torch.ones_like(pred_box[5 + i]), pred_box[5 + i])
                        class_loss += tclass_loss
                    else:
                        nclass_loss = self.MSELoss(torch.zeros_like(pred_box[5 + i]), pred_box[5 + i])
                        class_loss += nclass_loss
                print('# # # class loss : ', class_loss)

                loss = (xy_loss + wh_loss + obj_loss + nobj_loss + class_loss)
                print('# # # one batch loss : ', loss)
                total_loss.append(loss)

        total_loss = torch.stack(total_loss)
        print('total_loss before sum : ', total_loss)
        total_loss = total_loss.sum()
        print('final loss : ', total_loss)
        return total_loss

    def nencode_cls(self, prediction, target_idx, grid_size):
        npred_cls = []
        for nobj_i in range(grid_size):
            for nobj_j in range(grid_size):
                for _, p_idx in enumerate(target_idx):
                    i, j = p_idx
                    if nobj_i != i and nobj_j != j:
                        num_box1_cls = torch.unsqueeze(prediction[10:30, nobj_i, nobj_j], 0)
                        num_box2_cls = torch.unsqueeze(prediction[10:30, nobj_i, nobj_j], 0)
                        npred_cls_ = torch.cat((num_box1_cls, num_box2_cls), axis= 0)
                        npred_cls.append(npred_cls_)
                        break
        npred_cls = torch.cat(npred_cls)
        return npred_cls

    def nencode_box(self, prediction, target_idx, grid_size):
        npred_boxes = []
        # mask = np.argmax(a)
        # a[mask]
        # pred[mask]
        # pred[torch.logical_not(mask)]
        # prediction[0:5, target_idx]
        for nobj_i in range(grid_size):
            for nobj_j in range(grid_size):
                for _, p_idx in enumerate(target_idx):
                    i, j = p_idx
                    if nobj_i != i and nobj_j != j:
                        num_box1 = prediction[0:5, nobj_i, nobj_j]
                        num_box2 = prediction[5:10, nobj_i, nobj_j]
                        encode_box1 = self.make_box(num_box1)
                        encode_box2 = self.make_box(num_box2)
                        npred_box = torch.cat((encode_box1, encode_box2), axis=0)
                        npred_boxes.append(npred_box)
                        break
        npred_boxes = torch.cat(npred_boxes)
        return npred_boxes

    def make_box(self, box):
        x1 = box[0] 
        y1 = box[1] 
        width = box[2] 
        height = box[3] 
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

    def encode_cls(self, prediction, target_idx):
        for i, p_idx in enumerate(target_idx):
            num_box1_cls = torch.unsqueeze(prediction[10:30, p_idx[0], p_idx[1]], 0)
            num_box2_cls = torch.unsqueeze(prediction[10:30, p_idx[0], p_idx[1]], 0)
            pred_cls = torch.cat((num_box1_cls, num_box2_cls), axis=0)
        return pred_cls

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