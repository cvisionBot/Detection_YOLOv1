import cv2
import glob
import numpy as np

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class PreDataset(Dataset):
    def __init__(self, transforms, path=None, files_list=None, grid_size=7, num_boxes=2, num_classes=20):
        super(PreDataset, self).__init__()
        '''
        Data -> sample jpg + sample txt(cls, cx, cy, width, height)
        '''
        if path:
            self.imgs = glob.glob(path+'/*.jpg')
        if files_list:
            with open(files_list, 'r') as f:
                self.imgs = f.read().splotlines()
        self.transforms = transforms
        self.S = grid_size
        self.B = num_boxes
        self.C = num_classes

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img_file = self.imgs[index]
        img = cv2.imread(img_file)
        boxes = self.load_annotations(img_file, img_shape)
        transformed = self.transforms(image=img, bboxes=boxes)
        x_input = transformed['image']
        y_output = self.make_yolo_format(transformed['bboxes'], self.S, self.B, self.C)
        return x_input, y_output
    
    def load_annotations(self, img_file, img_shape):
        img_h, img_w, _ = img_shape
        annotations_file = img_file.replace('.jpg', 'txt')
        boxes = np.zeros((0, 5))
        with open(annotations_file, 'r') as f:
            annotations = f.read().splitlines()
            for annot in annotations:
                cid, cx, cy, w, h = map(float, annot.split(' '))
                x1 = (cx - w/2) * img_w
                y1 = (cy - h/2) * img_h
                w = w * img_w
                h = h * img_h
                annotation = np.array([[x1, y1, w, h, cid]])
                boxes = np.append(boxes, annotation, axis=0)
        return boxes
    
    def make_yolo_format(self, boxes, S, B, C):
        yolo_output = np.zeros((S, S, C + 5 * B))
        '''
            Box 하나를 가져와서 yolo output format으로 변경 [ 7 x 7 x 25]
            [7 x 7 x 20] = class probability
            [7 x 7 x 4] = boxes offset
            [7 x 7 x 1] = class info

        '''
        for box in boxes:


        yolo_output = torch.from_numpy(yolo_output)
        return yolo_output

class YoloFormat(pl.LightningDataModule):
    def __init__(self, train_list, val_list, workers, train_transforms, val_transforms, batch_size=None):
        super(YoloFormat, self).__init__()
        self.train_list = train_list
        self.val_list = val_list
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms
        self.batch_size = batch_size
        self.workers = workers

    def train_dataloader(self):
        return DataLoader(PreDataset(
            transforms=self.train_transforms,
            files_list=self.train_list),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.workers > 0,
            pin_memory=self.workers>0  
        )

    def val_dataloader(self):
        return DataLoader(PreDataset(
            transforms=self.val_transforms,
            files_list=self.val_list),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.workers,
            persistent_workers=self.workers > 0,
            pin_memory=self.workers > 0,
        )

if __name__ == '__main__':
    import albumentations
    import albumentations.pytorch
    
    train_transforms = albumentations.Compose([
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(),
        albumentations.RandomResizedCrop(320, 320, (0.8, 1)),
        albumentations.Normalize(0, 1),
        albumentations.pytorch.ToTensorV2(),
    ], bbox_params=albumentations.BboxParams(format='coco', min_visibility=0.1))

    loader = DataLoader(PreDataset(
        transforms=train_transforms, files_list='testfile_dir.txt'),
        batch_size=8, shuffle=True)
        
    for batch, sample in enumerate(loader):
        imgs = sample['img']
        annots = sample['annot']
    