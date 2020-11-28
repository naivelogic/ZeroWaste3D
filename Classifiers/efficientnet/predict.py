import json
from PIL import Image
import torch
from torchvision import transforms

from efficientnet_pytorch import EfficientNet
model = EfficientNet.from_pretrained('efficientnet-b0')

"""
ROOT_PATH = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v2/all_crops/'
learn_inf = load_learner(ROOT_PATH+'models/export_densenet161_test_small_classifier_22_crops_112520')

image_test = open_image("/mnt/zerowastepublic/csiro_ds2_outputs/ds2_outputs/cameratraps_classifier_baselogdir/crops/STC_0333.JPG___crop00_mdv1.0.jpg")
img_test_prediction = learn_inf.predict(image_test)
print("Results:")
print(str(img_test_prediction[0]))
print(img_test_prediction)
print(max(img_test_prediction[2]))
"""

#/home/redne/ZeroWaste3D/Detection/Classifier/repos/classifier2/savemodel/trained_model1.pth
#m = torch.load('/home/redne/ZeroWaste3D/Detection/Classifier/repos/classifier2/savemodel/trained_model1.pth')
m = torch.load('../simple_classifier2/model_best.pth.tar')
model.load_state_dict(m['state_dict'])


# Preprocess image
tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
test_img = '/mnt/omreast_users/phhale/csiro_trashnet/datasets/crop_ds0/v1/ds_x1/test/S_cup/csiro_real_ds0_000063.jpg'
img = tfms(Image.open(test_img)).unsqueeze(0)
print(img.shape) # torch.Size([1, 3, 224, 224])


#labels_map = json.load(open('demo_example/label_map.txt'))
#labels_map = [labels_map[str(i)] for i in range(1000)]

class_indices = {'D_lid': 0, 'H_beveragebottle': 1, 'M_beveragecan': 2, 'S_cup': 3}
labels_map=list(class_indices.keys())

# Classify
model.eval()
with torch.no_grad():
    outputs = model(img)

# Print predictions
print("Classifying Image")
print('-----')
for idx in torch.topk(outputs, k=3).indices.squeeze(0).tolist():
    prob = torch.softmax(outputs, dim=1)[0, idx].item()
    print('{label:<75} ({p:.2f}%)'.format(label=labels_map[idx], p=prob*100))