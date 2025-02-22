# Autonomous Driving

Created by **Jason Abi Chebli**  
© 2025 Jason Abi Chebli. All rights reserved.

## Code Authors
Binh Tran, Jason Abi Chebli, Meshach Kumar, RVSS

## Description
<div style="text-align: justify;">This project involves the development of an autonomous delivery robot capable of navigating Australian roads in both urban and rural environments. Using deep learning techniques, the system processes RGB camera images to generate steering commands, enabling the robot to drive autonomously while adhering to road rules, including stopping at designated stop signs. The project follows a structured workflow of data collection, model training, and deployment on a mobile robot.</div>

## Challenge
<div style="text-align: justify;">The task requires deploying a deep neural network on a mobile robot to process images and produce real-time steering commands. Participants must collect and label data, design and train a neural network, and test their models before deploying them on the robot. The competition consists of time trials on an unknown track, with penalties for errors such as leaving the road or failing to stop at a stop sign. The fastest and most accurate robots advance to the finals, where additional challenges are introduced.</div>

## Collecting Data
To ensure robust model training, we collected approximately 3,000 images by driving at a reduced speed in the competition room during the daytime. The dataset includes various road types:
- Straight urban roads
- Straight rural roads
- Sharp bends in urban roads
- Sharp bends in rural roads
- Slight turns in urban roads
- Slight turns in rural roads
- Edge-case roads, including train tracks, skid marks, and potholes

[Collecting Data Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/collect.py)  
[Dataset](https://github.com/jabichebli/autonomousDriving/tree/main/data)

## Training the Convolutional Neural Network
To improve generalization and robustness, we applied several data augmentation techniques to the training set:
- Random saturation color jitter
- Random contrast color jitter
- Random brightness color jitter
- Random grayscale conversion
- Random erasing (horizontal and vertical bars)
- Image resizing
- Data normalization

We categorized steering commands into seven distinct groups:
- Sharp left
- Left
- Slight left
- Straight
- Slight right
- Right
- Sharp right

<div style="text-align: justify;">
To balance the dataset and ensure fair training across all categories, we performed random sampling with replacement, effectively doubling the training dataset.

The neural network architecture consists of three convolutional layers, incorporating batch normalization, ReLU activation, and dropout for regularization.

We trained the model for 400 epochs, achieving an accuracy of **~75%**. 
</div>

<img src="https://github.com/jabichebli/autonomousDriving/blob/main/results/Accuracy_Curve.jpg" width="40%"> <img src="https://github.com/jabichebli/autonomousDriving/blob/main/results/Loss_Curve.jpg" width="40%">
<img src="https://github.com/jabichebli/autonomousDriving/blob/main/results/Confusion_Matrix.jpg" width="40%">

[Train Neural Network Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/train_net.py)

## Stop Sign Detection and Deployment on Raspberry Pi Penguin
<div style="text-align: justify;">To detect stop signs, we implemented color thresholding within a specific region of the camera's view. If a red area within the predefined range is detected at the lower boundary of the image and is of sufficient size, the robot will stop for 1.25 seconds before resuming movement, with a cooldown of 3 seconds between detections. </div>

Additionally, we incorporated dynamic speed control based on the road conditions:
- Maximum speed on straight paths
- Slightly reduced speed on slight turns
- Minimum speed on sharp turns

<div style="text-align: justify;">Despite operating at higher speeds than during training, the model performed well in real-world testing. Ultimately, we secured <strong>1st place</strong> in the <strong>Need4Speed Challenge</strong> at <strong>RVSS 2025</strong>.</div>

[Deployment Script](https://github.com/jabichebli/autonomousDriving/blob/main/scripts/deploy.py)

## Acknowledgements
Special thanks to [RVSS](https://www.rvss.org.au/) for providing the [base repository](https://github.com/rvss-australia/RVSS_Need4Speed).
