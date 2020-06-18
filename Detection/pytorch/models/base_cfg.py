
ML_WEIGHTS_PATH = '/home/redne/mnt/project_zero/pytorch/weights/'
DS01_CLASSES =  ['fork', 'spoon', 'knife', 'coffeeCup', 'clearCup']

# for making bounding boxes pretty
COLORS = ((255, 0, 0, 128), (0, 255, 0, 128), (0, 0, 255, 128),
          (0, 255, 255, 128), (255, 0, 255, 128), (255, 255, 0, 128))

MEANS = (104, 117, 123)

# model factory - https://github.com/CSAILVision/semantic-segmentation-pytorch/tree/master/models

# SSD300
class ssd_cfg:
    num_classes = 6 
    input_size= 300
    bbox_aspect_num= [4, 6, 6, 6, 4, 4]  
    feature_maps= [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    min_sizes = [30, 60, 111, 162, 213, 264]  
    max_sizes = [60, 111, 162, 213, 264, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    cuda= True
    train_labeled_file='/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/train_labels_dev.npy'
    val_labeled_file='/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/val_labels_dev.npy'
    img_dir='/home/redne/mnt/project_zero/project_zero/ds1/parsed/' #'data/imgs/',
    logs='checkpoints/logs.txt'
    color_mean=(104, 117, 123)
    variance= [0.1, 0.2],


ssd300_cfg = {
    'num_classes': 6,
    'input_size': 300,
    'base':300,
    'base_model':'vgg16_reducedfc.pth',
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'cuda': True,
    'train_labeled_file': '/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/train_labels_dev.npy',
    'val_labeled_file': '/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/val_labels_dev.npy',
    'img_dir': '/home/redne/mnt/project_zero/project_zero/ds1/parsed/',
    'logs': 'checkpoints/logs.txt',
    'color_mean': (104, 117, 123),
    'variance': ([0.1, 0.2],)
}

old_ssd_cfg = {
    'num_classes': 6, 
    'input_size': 300, 
    'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  
    'feature_maps': [38, 19, 10, 5, 3, 1],  
    'steps': [8, 16, 32, 64, 100, 300],  
    'min_sizes': [30, 60, 111, 162, 213, 264],  
    'max_sizes': [60, 111, 162, 213, 264, 315],  
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'cuda': True
}

class Args:
    
    backbone='resnet50'
    batch_size=1
    device='cuda:0'
    epochs=40
    img_dir='/home/redne/mnt/project_zero/project_zero/ds1/parsed/', #'data/imgs/',
    logs='checkpoints/logs.txt'
    long_size=1024
    lr=0.001
    lr_gamma=0.5
    lr_steps=[10, 15, 20, 25, 30]
    momentum=0.9
    print_freq=20
    resume=False
    save_freq=1
    train_labeled_file='/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/train_labels_dev.npy'
    val_labeled_file='/home/redne/mnt/project_zero/project_zero/ds1/experiments/data/val_labels_dev.npy'
    weight_decay=0.0001
    works=4



ARG_CONFIG ={
    'backbone':'resnet50',
    'batch_size':1,
    'device':'cuda',
    'epochs':40,
    'img_dir':'/mnt/zerowasteimages/project_zero/ds1/tester/', #'data/imgs/',
    'logs':'checkpoints/logs.txt',
    'long_size':1024,
    'lr':0.001,
    'lr_gamma':0.5,
    'lr_steps':[10, 15, 20, 25, 30],
    'momentum':0.9,
    'print_freq':20,
    'resume':False,
    'save_freq':1,
    'train_labeled_file':'data/tr_label.csv',
    'val_labeled_file':'data/val_label.csv',
    'weight_decay':0.0001,
    'works':4
}

# SSD300 CONFIGS
voc = {
    'num_classes': 21,
    'lr_steps': (8000, 10000, 12000),
    'max_iter': 12000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30, 60, 111, 162, 213, 264],
    'max_sizes': [60, 111, 162, 213, 264, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 201,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}
