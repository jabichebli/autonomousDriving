# Autonomous Driving

Created by **Jason Abi Chebli**  
© 2025 Jason Abi Chebli. All rights reserved.


## Code Authors
Binh Tran, Jason Abi Chebli, Meshach Kumar, RVSS

## Description
<div style="text-align: justify;">...</div>

## Challenge 
The task is to develop an autonomous delivery robot capable of safely and efficiently navigating Australian roads in both urban and rural environments. This involves deploying a deep neural network on a mobile robot to process RGB camera images and generate appropriate steering commands. Participants will follow a structured workflow, including data collection, labelling, network design, training, testing, and real-world deployment. The challenge culminates in a competition where teams navigate a designated track, encountering various road conditions and a stop sign that must be obeyed. The fastest and most accurate robots progress to the final round. Performance is scored based on delivery time, penalties for errors such as leaving the road or running a stop sign, and rewards for flawless navigation.

## Collecting Data
<div style="text-align: justify;">In our data collection method, we slowed dow the driving speed and collected a range of data during the day in the room the competition was held in. We ended up collecting around 3000 images. This data includes:</div>
- Straight Urban Road
- Straight Rural Road
- Sharp Bends Urban Road
- Sharp Bends Rural Road
- Slight Turns Urban Road
- Slight Turn Rural Road
- Edge Case roads including train tracks, skid marks, pot holes etc.

[Collecting Data Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/collect.py)
[Data](https://github.com/jabichebli/autonomousDriving/tree/main/data)

## Training the Convulutional Neural Network 
<div style="text-align: justify;">To make our system as robust as possible, the dataset we trained it on had the following transforms applied:</div>
-  Random Saturation Color Jitter
- Random Contrast Color Jitter
- Random Bridgness Color Jitter
- Random Gray scale 
- Random Erasing Horizontally and vertical bars
- Resizing the image
- Normalising the Data 

<div style="text-align: justify;"> We Split the classification of the steering into 7 different groups including:</div>
- sharp left 
- left 
- slight left 
- straight 
 -slight right
- right
- sharp right

<div style="text-align: justify;">To ensure the model is trained fairly on all cases, all cases were randomly sampled with replacement (2x train dataset) to have even weights.</div>


<div style="text-align: justify;">The neural network consisted of three layers, using backNorm, ReLU and Dropout.</div>


<div style="text-align: justify;">We found that training out model to 400 Epochs was a good fit. With accuracy of: __%</div>


<div style="text-align: justify;"></div>


[Train Neural Network Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/train_net.py)

## Colour Thresholding (Stop Sign Detection) and Deploying the Neural Network on Raspberry Pi Penguin
<div style="text-align: justify;">...</div>

[Deploy Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/deploy.py)

## Acknowledgement
Thanks to [RVSS](https://www.rvss.org.au/) who provided us with the [base repository](https://github.com/rvss-australia/RVSS_Need4Speed).

