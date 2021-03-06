import torch 
import numpy as np 
import cv2 
 
 
def collater(data): 
    """Data Loader에서 생성된 데이터를 동일한 shape으로 정렬해서 Batch로 전달 
 
    Args: 
        data ([dict]): albumentation Transformed 객체 
        'image': list of Torch Tensor len == batch_size, item shape = ch, h, w 
        'bboxes': list of list([x1, y1, w, h, cid]) 
 
    Returns: 
        [dict]: 정렬된 batch data. 
        'img': list of image tensor 
        'annot': 동일 shape으로 정렬된 tensor [x1,y1,w,h] format 
    """ 
    imgs = [s['image'] for s in data] 
    bboxes = [torch.tensor(s['bboxes'])for s in data] 
    batch_size = len(imgs) 
 
    max_num_annots = max(annots.shape[0] for annots in bboxes) 
 
    if max_num_annots > 0: 
        padded_annots = torch.ones((batch_size, max_num_annots, 5)) * -1 
        for idx, annot in enumerate(bboxes): 
            if annot.shape[0] > 0: 
                # To x1, y1, w, h 
                padded_annots[idx, :annot.shape[0], :] = annot 
    else: 
        padded_annots = torch.ones((batch_size, 1, 5)) * -1 
 
    return {'img': torch.stack(imgs), 'annot': padded_annots}