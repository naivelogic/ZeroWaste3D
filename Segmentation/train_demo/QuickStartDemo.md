# Quick start on training 

### Install and Setup

```
# pytorch install
pip install torch==1.4.0 torchvision==0.5.0

# install cython and fvcore
pip install cython; pip install -U 'git+https://github.com/facebookresearch/fvcore.git' 

# clone and install detectron repo 
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

# install remaining dependencies
pip install opencv-python pillow pycocotools matplotlib tqdm
```

## Usage

Train the model using `python training.py -[] `

pip install torch==1.2.0 torchvision==0.4.0
~/Yolact_fcos$ python train_net.py --config-file configs/Yolact/yfcos_ds2_r50_x0.yaml