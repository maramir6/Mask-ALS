from PIL import Image
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops_table
from dataset2 import LidarDataset2 
from engine import train_one_epoch, evaluate
import utils
import transforms as T
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage.io import imsave
import pandas as pd
import numpy as np
import torch
import os

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def get_model_instance_segmentation(num_classes, input_channel, hidden_layer=256):
    
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    #model.rpn.anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128),), aspect_ratios=((0.5, 1.0, 2.0),))
    
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels

    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)

    return model

def main(input_folder, output_folder):

    # our dataset has two classes only - background and person
    num_classes = 2
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes, input_channel=3)
    checkpoint = torch.load('E:/PYTORCH-MASK/model_final_2.pth')
    model.load_state_dict(checkpoint)
    model.eval()
    model.to(device)

    cpu_device = torch.device("cpu")
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Prediction:'

    dataset = LidarDataset2(input_folder)

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, collate_fn=utils.collate_fn)

    for image, targets, file_names in metric_logger.log_every(data_loader, 100, header):
        
        image = list(img.to(device) for img in image)
        torch.cuda.synchronize()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]

        index = 0
        for output in outputs:
        	file_name = file_names[index]

        	print(output)
        	masks = output["masks"].detach().numpy()
        	output = np.zeros((masks.shape[2],masks.shape[3]))
        	id_t = 1

        	for i in range(masks.shape[0]):
        		indice = np.where(masks[i,0,:,:]>0.75)
        		output[indice] = id_t
        		id_t = id_t + 1

        	mask_final = output > 0

        	props = regionprops_table(label(mask_final), properties=('label', 'area')) 
        	df = pd.DataFrame(props)
        	area = df['area'].values
        	min_area = 0.5*np.median(area)
        	mask_final = remove_small_objects(mask_final, int(min_area))
        	mask_final = remove_small_holes(mask_final, int(min_area))
        	output = label(mask_final)
        	mpimg.imsave(output_folder + file_name, output, cmap=plt.cm.nipy_spectral)
        	index = index + 1

if __name__ == '__main__':

	input_folder = 'E:/RODALES/PARCELA_COSECHA/IMAGES/'
	output_folder = 'E:/RODALES/PARCELA_COSECHA/LABELS/'
	
	main(input_folder, output_folder)