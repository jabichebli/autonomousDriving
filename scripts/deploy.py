#!/usr/bin/env python3
import time
import click
import math
import cv2
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import argparse
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from scipy.ndimage import label
script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))
from pibot_client import PiBot


def stop_sign_detection(im, min_area=500, thres = 100):
    ### ---- Detect Stop Sign ---- ###
   
    # Convert the image to HSV color space for color segmentation
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)


    # Define refined HSV ranges for more precise pure red detection
    lower_red1 = np.array([0, 150, 100])  # Higher saturation and brightness to exclude orange
    upper_red1 = np.array([1, 255, 255])


    lower_red2 = np.array([170, 150, 100])  # Focus on the deeper red tones
    upper_red2 = np.array([180, 255, 255])


    # Create masks for the refined red range
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)


    # OPTIONAL: Remove small blobs/noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    red_mask_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    # Label connected components (blobs)
    labeled_array, num_features = label(red_mask_cleaned,structure=[[1,1,1],[1,1,1],[1,1,1]])
   
    # Find object slices and filter small blobs
    for i in range(1,num_features+1):
        blob_size = np.sum(labeled_array == i)
        if blob_size < min_area:
            labeled_array[labeled_array == i] = 0  # Remove small blobs
   
    # Relabel after removal
    labeled_array[labeled_array > 0] = 1
    if not np.any(labeled_array):
        return False, labeled_array
    else:
        bottom_line = np.max(np.where(labeled_array)[0])
        if bottom_line > thres:
            return True, labeled_array
        else:
            return False, labeled_array


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 5)


        self.pool = nn.MaxPool2d(2, 2)


        self.fc1 = nn.Linear(512, 64)
        self.fc2 = nn.Linear(64, 7)


        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(32)
        self.drop = nn.Dropout()




    def forward(self, x):
        #extract features with convolutional layers
        x = self.pool(self.drop(self.relu(self.conv1(x))))
        x = self.pool(self.drop(self.relu(self.conv2(x))))
        x = self.pool(self.drop(self.relu(self.conv3(x))))
        x = self.norm(x)
        x = torch.flatten(x, 1) # flatten all dimensions except batch
       
        #linear layer for classification
        x = self.fc1(x)
        x = self.drop(self.relu(x))
        x = self.fc2(x)
       
        return x
   
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Resize((60, 60)),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PiBot client')
    parser.add_argument('--ip', type=str, default='localhost', help='IP address of PiBot')
    args = parser.parse_args()


    bot = PiBot(ip=args.ip)


    # stop the robot
    bot.setVelocity(0, 0)


    #INITIALISE NETWORK HERE


    #LOAD NETWORK WEIGHTS HERE


    #countdown before beginning
    print("Get ready...")
    time.sleep(1)
    print("3")
    time.sleep(1)
    print("2")
    time.sleep(1)
    print("1")
    time.sleep(1)
    print("GO!")


    net = Net()
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #net.load_state_dict(torch.load('steer_net_best700.pth', map_location=device))
    net.load_state_dict(torch.load('steer_net_bestest400.pth', weights_only=True))
    net.eval()
    angles = [-0.50, -0.30, -0.15, 0, 0.15, 0.30, 0.50]
    try:
        angle = 0
        # Check for stop signs
        detected = False
        cd_start_time = time.time()


        while True:
            # get an image from the the robot
            im = bot.getImage()


            #Apply any necessary image transforms
            im = im[120:, :, :]


            # Do Stop Sign Detection
            detected, mask = stop_sign_detection(im,350, 90)
            while True:
                cv2.imshow('image', (mask*255).astype(np.uint8))
                if cv2.waitKey(1):
                    break


            #Apply any necessary image transforms
            im = transform(im)
            im = torch.unsqueeze(im,0)


            #TO DO: check for stop signs?
            outputs = net(im)
            _, predictions = torch.max(outputs, 1)
            pred = predictions[0]
            angle = angles[pred]
            print(angle)


            if pred == 3:
                Kd = 26 #base wheel speeds, increase to go faster, decrease to go slower
                Ka = 39 #how fast to turn when given an angle
            elif pred == 2 or pred == 4:
                Kd = 24 #base wheel speeds, increase to go faster, decrease to go slower
                Ka = 36 #how fast to turn when given an angle
            elif pred == 1 or pred == 5:
                Kd = 18 #base wheel speeds, increase to go faster, decrease to go slower
                Ka = 27 #how fast to turn when given an angle
            else:
                Kd = 14 #base wheel speeds, increase to go faster, decrease to go slower
                Ka = 21 #how fast to turn when given an angle


            if detected and ((time.time() - cd_start_time) > 3):
                bot.setVelocity(0, 0) # stop the vehicle
                print("Stop sign detected")
                # while True:
                #     cv2.imshow('image', mask)
                #     if cv2.waitKey(1) & 0xFF == ord('q'):
                #         break
                time.sleep(1.25)


                detected = False #reset detected flag
                cd_start_time = time.time()


            else:
                left  = int(Kd + Ka*angle)
                right = int(Kd - Ka*angle)
                bot.setVelocity(left, right)
               
           
    except KeyboardInterrupt:    
        bot.setVelocity(0, 0)

