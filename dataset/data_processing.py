import xml.etree.ElementTree as elemTree
import glob
import tqdm
import os

class VOC:
    def __init__(self, img_path, xml_path, names_file):
        super().__init__()
        self.xml_path = xml_path
        self.labels = self.read_name(names_file)
        self.img_path = img_path
    
    
    def read_name(self, names_file):
        if not os.path.exists(names_file):
            print("no folder : ", names_file)
            return os.error
        labels = []
        with open(names_file, "r") as f:
            labels = f.read().splitlines()
            f.close()
        return labels
    
    
    def make_label(self, names_file):
        if not os.path.exists(self.xml_path):
            print("no folder : ", self.xml_path)
            return os.error
        labels = []
        annots_xml_file = glob.glob(os.path.join(self.xml_path, '*.xml'))
        for annot_file in tqdm.tqdm(annots_xml_file):
            tree = elemTree.parse(annot_file)
            objects_info = tree.findall('object')
            for object_info in objects_info:
                labels.append(object_info.findtext('name'))
        labels = sorted(list(set(labels)))
        with open(names_file, "w") as f:
            f.writelines('\n'.join(labels))
            f.close()
        return labels


    def save_train_val_set(self, save_train_set, save_val_set, ratio=0.5):
        import random
        if not os.path.exists(self.img_path):
            print("no folder : ", self.img_path)
        imgs_file = glob.glob(os.path.join(self.img_path, '*.jpg'))
        random.shuffle(imgs_file)
        train_set = imgs_file[:int(len(imgs_file)*ratio)]
        val_set = imgs_file[int(len(imgs_file)*ratio):]
        if save_train_set:
           with open(save_train_set, "w") as f:
                f.writelines('\n'.join(train_set))
                f.close()
        
        if save_val_set:
           with open(save_val_set, "w") as f:
                f.writelines('\n'.join(val_set))
                f.close()


    def save_txt(self, dst_path):
        # cls, cx, cy, w, h (using normalized data)
        if not os.path.exists(self.xml_path):
            print("no folder : ", self.xml_path)
            return os.error
        if not os.path.exists(dst_path):
            print("make dir : ", dst_path)
            os.makedirs(dst_path)
        annots_xml_file = glob.glob(os.path.join(self.xml_path, '*.xml'))
        for annot_file in tqdm.tqdm(annots_xml_file):
            tree = elemTree.parse(annot_file)
            img_name = tree.findtext('filename')
            img_w, img_h = float(tree.find('size').findtext('width')), float(tree.find('size').findtext('height'))
            objects_info = tree.findall('object')
            yolo_form_info = []
            for object_info in objects_info:
                box_info = object_info.find('bndbox')
                obj_name = object_info.findtext('name')
                c = self.labels.index(obj_name)
                w = (float(box_info.findtext('xmax')) - float(box_info.findtext('xmin'))) / img_w
                h = (float(box_info.findtext('ymax')) - float(box_info.findtext('ymin'))) / img_h
                cx = (float(box_info.findtext('xmin')) / img_w) + (w/2)
                cy = (float(box_info.findtext('ymin')) / img_h) + (h/2)
                yolo_form_info.append('{} {} {} {} {}'.format(c, cx, cy, w, h))
                
            yolo_form_file = os.path.join(dst_path, img_name.replace('.jpg', '.txt'))
            with open(yolo_form_file, "w") as f:
                f.writelines('\n'.join(yolo_form_info))
                f.close()

if __name__ == "__main__":
<<<<<<< HEAD
    voc_pars = VOC(img_path = "../dataset/voc_train/JPEGImages", xml_path="../dataset/voc_train/Annotations", names_file="./dataset/names/pascal_voc.txt")
    label = voc_pars.make_label(names_file="./dataset/names/pascal_voc.txt")
    print(label)
    # voc_pars.save_txt(dst_path="/mnt/voc_train/JPEGImages")
    # voc_pars.save_train_val_set("/mnt/voc_train/train_list.txt", "/mnt/voc_train/val_list.txt", ratio=0.7) # train set ratio
=======
    voc_pars = VOC(img_path = "../dataset/voc_train/JPEGImages", xml_path="../dataset/voc_train/Annotations", names_file="./dataset/names/pascal_voc.txt", dst_path="../dataset/voc_train/JPEGImages")
    voc_pars.save_txt()
    label = voc_pars.make_label()
    voc_pars.save_train_val_set("../dataset/voc_train/train_list.txt", "../dataset/voc_train/val_list.txt", ratio=0.5) # train set ratio
    
>>>>>>> 69b52c12216a0a20a31415444297d887594a75f0
