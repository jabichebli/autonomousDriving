import numpy as np
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import torch
import cv2
from glob import glob
from os import path

class SteerDataSet(Dataset):
    
    def __init__(self,root_folder,img_ext = ".jpg" , transform=None):
        self.root_folder = root_folder
        self.transform = transform        
        self.img_ext = img_ext        
        self.filenames = glob(path.join(self.root_folder,"*" + self.img_ext))            
        self.totensor = transforms.ToTensor()
        self.class_labels = ["sharp left",
                             "left",
                             "slight left",
                             "straight",
                             "slight right",
                             "right",
                             "sharp right"]
        
    def __len__(self):        
        return len(self.filenames)
    
    def __getitem__(self,idx):
        f = self.filenames[idx]        
        img = cv2.imread(f)
        
        img = img[120:, :, :]
        
        if self.transform == None:
            img = self.totensor(img)
        else:
            img = self.transform(img)   
        
        steering = path.basename(f).split(self.img_ext)[0][7:]
        steering = float(steering)


        if steering <= -0.5:
            steering_cls = 0
        elif steering <= -0.3:
            steering_cls = 1
        elif steering <= -0.1:
            steering_cls = 2
        elif steering == 0:
            steering_cls = 3
        elif steering <= 0.2:
            steering_cls = 4
        elif steering <= 0.4:
            steering_cls = 5
        else:
            steering_cls = 6

                      
        return img, steering_cls