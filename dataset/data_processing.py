from types import LambdaType
import xml.etree.ElementTree as elemTree
import glob
import tqdm
import os

class VOC:
    def __init__(self, xml_path, names_file, dst_path):
        super().__init__()
        self.xml_path = xml_path
        self.labels = self.read_name(names_file)
        self.dst_path = dst_path
    
    
    def read_name(self, names_file):
        if not os.path.exists(names_file):
            print("no file : ", names_file)
            return os.error
        labels = []
        with open(names_file, "r") as f:
            labels = f.read().splitlines()
            f.close()
        return labels
    
    
    def make_label(self):
        if not os.path.exists(self.xml_path):
            print("no file : ", self.xml_path)
            return os.error
        labels = []
        annots_xml_file = glob.glob(os.path.join(self.xml_path, '*.xml'))
        for annot_file in tqdm.tqdm(annots_xml_file):
            tree = elemTree.parse(annot_file)
            objects_info = tree.findall('object')
            for object_info in objects_info:
                labels.append(object_info.findtext('name'))
        labels = sorted(list(set(labels)))
        return labels


    def save_txt(self):
        if not os.path.exists(self.xml_path):
            print("no file : ", self.xml_path)
            return os.error
        if not os.path.exists(self.dst_path):
            print("no file : ", self.dst_path)
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
                x1 = float(box_info.findtext('xmin')) / img_w
                y1 = float(box_info.findtext('ymin')) / img_h
                w = (float(box_info.findtext('xmax')) - float(box_info.findtext('xmin'))) / img_w
                h = (float(box_info.findtext('ymax')) - float(box_info.findtext('ymin'))) / img_h
                yolo_form_info.append('{} {} {} {} {}\n'.format(c, x1, y1, w, h))
                
            yolo_form_file = os.path.join(self.dst_path, img_name.replace('.jpg', '.txt'))
            with open(yolo_form_file, "w") as f:
                for line in yolo_form_info:
                    f.write(line)
                f.close()

if __name__ == "__main__":
    voc_pars = VOC(xml_path="../dataset/voc_train/Annotations", names_file="./dataset/names/pascal_voc.txt", dst_path="../dataset/voc_train/annot_txt")
    # voc_pars.save_txt()
    label = voc_pars.make_label()
    print(label)