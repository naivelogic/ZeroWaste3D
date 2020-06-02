import numpy as np
import matplotlib.pyplot as plt 
import cv2  
import torch
import time
from utils.data_processor import VOC_DataTransform as DataTransform
import torch.nn as nn

CLASS_NUM = 6

class SSDPredictShow(nn.Module):

    def __init__(self, eval_categories, net, device, TTA=True, image_size=300):
        super(SSDPredictShow, self).__init__() 
        print(device)
        self.eval_categories = eval_categories 
        self.net = net.to(device).eval() 
        self.device = device
        self.TTA=TTA

        color_mean = (104, 117, 123) 
        input_size = image_size 
        self.transform = DataTransform(input_size, color_mean) 

    def show(self, image_file_path, data_confidence_level):
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        img = cv2.imread(image_file_path)  
        height, width, channels = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        phase = "val"
        img_transformed, boxes, labels = self.transform(img, phase, "", "")
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)
        print(img.size())
        
        x = img.unsqueeze(0)

        detections = self.net(x)

        # confidence_level
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections.cpu().detach().numpy()

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):
            if (find_index[1][i]) > 0:  
                sc = detections[i][0]  
                bbox = detections[i][1:] * [width, height, width, height]

                lable_ind = find_index[1][i]-1
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def ssd_predict2(self, image_file_path, data_confidence_level=0.5):
        """
        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        img = cv2.imread(image_file_path) 
        height, width, channels = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        phase = "val"
        img_transformed, boxes, labels = self.transform(img, phase, "", "")
        img = torch.from_numpy(
            img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)

        x = img.unsqueeze(0)
        
        with torch.no_grad():
            detections = self.net(x)

        # confidence
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        try:
            detections = detections.cpu().detach().numpy()
        except:
            detections = detections.detach().numpy()

        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):
            if (find_index[1][i]) > 0: 
                sc = detections[i][0]  
                detections[i][1:] *= [width, height, width, height]
                lable_ind = find_index[1][i]-1
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return detections, pre_dict_label_index
    
    def ssd_inference(self, dataloader, all_boxes, data_confidence_level=0.05):
        """
        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """
        
        empty_array = np.transpose(np.array([[],[],[],[],[]]), (1,0))
        iii=0 # image number
        width = 300
        height = 300
        for img, _ in dataloader:
            num_batch = len(img)
            self.net.eval().to(self.device)
            tick = time.time()
            with torch.no_grad():
                x = img.to(self.device)  
                detections = self.net(x)
                
            tock = time.time()
                        
            predict_bbox = []
            pre_dict_label_index = []
            scores = []
            detections = detections.cpu().detach().numpy()
            print(detections.shape)

            took = time.time()
            for batch, detection in enumerate(detections):
                for cls in range(CLASS_NUM):
                    box = []
                    for j,pred in enumerate(detection[cls]):
                        if pred[0] > data_confidence_level:
                            pred[1:] *= width
                            box.append([pred[0],pred[1],pred[2],pred[3],pred[4]])
                    if not box == []:
                        all_boxes[cls][iii*num_batch + batch] = box
                    else:
                        all_boxes[cls][iii*num_batch + batch] = empty_array
                    
            teek = time.time()
            print("iter:", iii)            
            iii += 1
            
            print("sort boxes. detection was {} and post took {} and allboxappend took {}".format(tock-tick, took-tock, teek-took))
            
        return all_boxes
    
    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        num_classes = len(label_names)  
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()

        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()

        for i, bb in enumerate(bbox):

        
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]  

            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            currentAxis.add_patch(plt.Rectangle(xy, width, height, fill=False, edgecolor=color, linewidth=2))

            currentAxis.text(xy[0], xy[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})

class Detect_Flip(nn.Module):
    def __init__(self, conf_thresh=0.01, top_k=200, nms_thresh=0.45, TTA=True, softnms=False):
        super(Detect_Flip, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
        self.conf_thresh = conf_thresh
        self.top_k = top_k
        self.nms_thresh = nms_thresh
        self.TTA = TTA
        self.softnms = softnms
        if TTA:
            print("test time flip is ON.")
        else:
            print("test time flip is OFF.")
        print("nms thresh is :", nms_thresh)
        print("soft nms is : ", softnms)
            
    def forward(self, loc_data, conf_data, loc_data2, conf_data2, dbox_list):
        
        num_batch = loc_data.size(0)
        num_dbox = loc_data.size(1)
        num_classes = conf_data.size(2)
        
        # conf + softmax
        conf_data = self.softmax(conf_data)
        conf_data2 = self.softmax(conf_data2)
        
        # [batch, class, topk, 5]
        output = torch.zeros(num_batch, num_classes, self.top_k, 5)
        
        # conf_data
        # [batch, 8732, classes]
        conf_preds = conf_data.transpose(2, 1)
        # [batch, classes, 8732]
        conf_preds2 = conf_data2.transpose(2, 1)
        
        # batch
        for i in range(num_batch):
            # 1. Loc DBox BBox
            decoded_boxes = decode(loc_data[i], dbox_list)
            decoded_boxes2 = decode(loc_data2[i], dbox_list)
            
            # conf
            conf_scores = conf_preds[i].clone()
            conf_scores2 = conf_preds2[i].clone()
            
            # class/NMS
            for cl in range(1, num_classes): # 背景は飛ばす。
                
                c_mask = conf_scores[cl].gt(self.conf_thresh) # gt=greater than
                c_mask2 = conf_scores2[cl].gt(self.conf_thresh) # gt=greater than
                # index mask
                # thresh
                # c_mask
                
                scores = conf_scores[cl][c_mask]
                scores2 = conf_scores2[cl][c_mask2]               
                               
                if scores.nelement() == 0:
                    continue
                    
                    
                # cmask box
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                l_mask2 = c_mask2.unsqueeze(1).expand_as(decoded_boxes2)
                
                boxes = decoded_boxes[l_mask].view(-1, 4) # reshape to [boxnum, 4]
                boxes2 = decoded_boxes2[l_mask2].view(-1, 4) # reshape to [boxnum, 4]
                
                # boxes2 are flipped.. fix that.
                tmpbox = boxes2
                boxes2[:, 0] = 1 - tmpbox[:, 2]
                boxes2[:, 2] = 1 - tmpbox[:, 0]
                              
                # concat boxes and score
                if self.TTA:
                    boxes = torch.cat((boxes, boxes2), 0)
                    scores = torch.cat((scores, scores2), 0)
                
                #print(boxes.shape)
                #print(scores.shape)
                
                if not self.softnms:
                    # 3. NMS
                    ids, count = nms(boxes, scores, self.nms_thresh, self.top_k)
                    #count = len(ids)
                    ##torch.cat(tensors, dim=0, out=None) → Tensor
                    output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)
                
                # 4. soft-nms
                else:
                    boxes, scores = softnms(boxes, scores)                
                    if boxes == []:
                        continue                   
                    boxes = torch.stack(boxes)
                    scores = torch.stack(scores)

                    output[i, cl, :len(scores)] = torch.cat((scores.unsqueeze(1), boxes), 1)
                
        return output      

class SSDPredictShowFlip(nn.Module):

    def __init__(self, eval_categories, net, device, TTA=True, softnms=False):
        super(SSDPredictShowFlip, self).__init__()  
        print(device)
        self.eval_categories = eval_categories  
        self.net = net.to(device)  # SSD
        self.device = device
        self.TTA =TTA

        color_mean = (104, 117, 123)  # (BGR)
        input_size = 300  # 300×300
        self.transform = DataTransform(input_size, color_mean)  
        
        self.Det = Detect_Flip(TTA=TTA, softnms=softnms).to(self.device).eval()

    def show(self, image_file_path, data_confidence_level):
        
        rgb_img, predict_bbox, pre_dict_label_index, scores = self.ssd_predict(
            image_file_path, data_confidence_level)

        self.vis_bbox(rgb_img, bbox=predict_bbox, label_index=pre_dict_label_index,
                      scores=scores, label_names=self.eval_categories)

    def ssd_predict(self, image_file_path, data_confidence_level=0.5):
        """
        Returns
        -------
        rgb_img, true_bbox, true_label_index, predict_bbox, pre_dict_label_index, scores
        """

        
        img = cv2.imread(image_file_path)  
        height, width, channels = img.shape 
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        
        phase = "val"
        img_transformed, boxes, labels = self.transform(img, phase, "", "")
           
        img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1).to(self.device)

        
        x = img.unsqueeze(0)
        with torch.no_grad():
            detections = self.net(x)
        
        
        ## Flip inference
        x_flip = torch.flip(img, [2])
        x_flip = x_flip.unsqueeze(0)
        with torch.no_grad():
            detections_flip = self.net(x_flip)
        
        #print("check box: ", (detections[2]==detections_flip[2]).sum().numpy())
        
        ## Gather detections.
        detections_box = self.Det(detections[0], detections[1], detections_flip[0], detections_flip[1], detections[2].to(self.device))
        
        # confidence_level
        predict_bbox = []
        pre_dict_label_index = []
        scores = []
        detections = detections_box.cpu().detach().numpy()
        

        
        find_index = np.where(detections[:, 0:, :, 0] >= data_confidence_level)
        detections = detections[find_index]
        for i in range(len(find_index[1])):  
            if (find_index[1][i]) > 0:  
                sc = detections[i][0]  
                bbox = detections[i][1:] * [width, height, width, height]
                
                lable_ind = find_index[1][i]-1
                
                predict_bbox.append(bbox)
                pre_dict_label_index.append(lable_ind)
                scores.append(sc)

        return rgb_img, predict_bbox, pre_dict_label_index, scores

    def vis_bbox(self, rgb_img, bbox, label_index, scores, label_names):
        
        num_classes = len(label_names)
        colors = plt.cm.hsv(np.linspace(0, 1, num_classes)).tolist()


        plt.figure(figsize=(10, 10))
        plt.imshow(rgb_img)
        currentAxis = plt.gca()


        for i, bb in enumerate(bbox):

            
            label_name = label_names[label_index[i]]
            color = colors[label_index[i]]

            
            if scores is not None:
                sc = scores[i]
                display_txt = '%s: %.2f' % (label_name, sc)
            else:
                display_txt = '%s: ans' % (label_name)

            
            xy = (bb[0], bb[1])
            width = bb[2] - bb[0]
            height = bb[3] - bb[1]

            
            currentAxis.add_patch(plt.Rectangle(
                xy, width, height, fill=False, edgecolor=color, linewidth=2))

            
            currentAxis.text(xy[0], xy[1], display_txt, bbox={
                             'facecolor': color, 'alpha': 0.5})

            