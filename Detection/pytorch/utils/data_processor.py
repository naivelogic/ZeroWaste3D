"""
ZeroWaste Data Processor for Pytorch 
Dataset 1
"""
import numpy as np
import torch
import json, os, glob, cv2
from torch.utils.data import Dataset
from utils.data_augmentations import Compose, ConvertFromInts, ToAbsoluteCoords, PhotometricDistort, Expand, RandomSampleCrop, RandomMirror, ToPercentCoords, Resize, SubtractMeans

def make_datapath_list(rootpath):
    imgpath = glob.glob(rootpath + '*_rgb.jpg')
    annpath = glob.glob(rootpath + '*_bbox.json')
    return imgpath[:60], annpath[:60]
    #return imgpath, annpath


class Anno_json(object):
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, json_file, width, height):
        
        with open(json_file) as f:
            file = json.load(f)

            ret = []
            for _, row in enumerate(file):
                bndbox = []
                name = row['category']
                xmin, ymin, w, h = row['rect']
                xmax = xmin + w
                ymax = ymin + h            
                
                
                boxes = [xmin, ymin, xmax, ymax]
                for i, k in enumerate(boxes):
                    if i == 0 or i == 2:
                        val = float(k/width)
                    else:
                        val = float(k/height)

                    bndbox.append(val)
                                    
                label_idx = self.classes.index(name)
                bndbox.append(label_idx)
            
                ret += [bndbox]

            return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_ind], ... ]
                
class VOC_DataTransform():
    """
    Attributes
    ----------
    input_size : int
    color_mean : (B, G, R)

    """

    def __init__(self, input_size, color_mean):
        self.data_transform = {
            'train': Compose([
                ConvertFromInts(),
                ToAbsoluteCoords(),
                PhotometricDistort(),
                Expand(color_mean),  
                RandomSampleCrop(),  
                RandomMirror(),  
                ToPercentCoords(), 
                Resize(input_size), 
                SubtractMeans(color_mean)
            ]),
            'val': Compose([
                ConvertFromInts(),  
                Resize(input_size), 
                SubtractMeans(color_mean)
            ])
        }

    def __call__(self, img, phase, boxes, labels):
        return self.data_transform[phase](img, boxes, labels)




class VOCDataset(Dataset):
    """
    Inherit PyTorch's Dataset class
    Attributes
    ----------
    img_list: list
    anno_list: list
    phase: 'train' or 'test'
    transform: object
    transform_anno: object
    """

    def __init__(self, img_list, anno_list, phase, transform, transform_anno):
        self.img_list = img_list
        self.anno_list = anno_list
        self.phase = phase  
        self.transform = transform  
        self.transform_anno = transform_anno

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        
        im, gt, h, w = self.pull_item(index)
        return im, gt

    def pull_item(self, index):
        
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  
        height, width, channels = img.shape

        anno_file_path = self.anno_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        img, boxes, labels = self.transform(
            img, self.phase, anno_list[:, :4], anno_list[:, 4])

        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width

def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  
        targets.append(torch.FloatTensor(sample[1]))  

    imgs = torch.stack(imgs, dim=0)

    return imgs, targets