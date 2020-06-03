import os
from numba import git
import numpy as np
import torch
from torchvision.transforms import functional as F
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def slice_image(image, size):

    hsize =(size/2)
    x_n, y_n = image.shape
    xi, yi = x_n//hsize, y_n//hsize
    imgs = []
    coords = []

    for i in range(int(xi-1)):
        for j in range(int(yi-1)):
            crop = image[int(i*hsize):int(i*hsize+size),int(j*hsize):int(j*hsize+size)]
            imgs.append(crop)
            coords.append((int(i*hsize),int(j*hsize)))

    return imgs, coords

class LidarDataset(object):
    def __init__(self, image, size):
        
        self.imgs, self.coords = slice_image(image, size)
    
    def __getitem__(self, idx):

        img = self.imgs[idx]
        coord = self.coords[idx]
        img = F.to_tensor(np.array([img, img, img]))

        return img, coord

    def __len__(self):
        return len(self.imgs)
