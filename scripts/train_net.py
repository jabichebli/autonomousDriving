import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import os
import numpy as np
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from steerDS import SteerDataSet

#######################################################################################################################################
####     This tutorial is adapted from the PyTorch "Train a Classifier" tutorial                                                   ####
####     Please review here if you get stuck: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html                   ####
#######################################################################################################################################
torch.manual_seed(0)

#Helper function for visualising images in our dataset
def imshow(img):
    img = img / 2 + 0.5 #unnormalize
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0))
    rgbimg = npimg[:,:,::-1]
    plt.imshow(rgbimg)
    plt.show()

'''
# Define transformations
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # Random crop for augmentation
    transforms.RandomHorizontalFlip(),  # Random horizontal flip
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize to fixed size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
script_path = os.path.dirname(os.path.realpath(__file__))
# Load full dataset without transforms (we'll apply them later)
full_dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'))

# Split dataset into train and test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

# Apply different transforms using dataset wrappers
train_dataset.dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), transform=train_transform)
test_dataset.dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), transform=test_transform)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Check dataset sizes
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")

exit()
'''
#######################################################################################################################################
####     SETTING UP THE DATASET                                                                                                    ####
#######################################################################################################################################

batch_size = 32

#transformations for raw images before going to CNN
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomChoice([
        transforms.ColorJitter(saturation=0.5),
        transforms.ColorJitter(contrast=0.5),
        transforms.ColorJitter(brightness=0.5)], [1, 1, 1]),
    transforms.RandomGrayscale(0.1),
    transforms.RandomApply([
        transforms.RandomErasing(0.5,(0.05, 0.15),(3.3, 3.3)),
        transforms.RandomErasing(0.5,(0.05, 0.15),(0.3, 0.3))],0.8),
    transforms.Resize((60, 60)),  # Random crop for augmentation
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((60, 60)),  # Random crop for augmentation
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

script_path = os.path.dirname(os.path.realpath(__file__))

# Load full dataset without transforms (we'll apply them later)
full_dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), '.jpg')

# Split dataset into train and test
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_ds, val_ds = random_split(full_dataset, [train_size, test_size])

# Apply different transforms using dataset wrappers
train_ds.dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), '.jpg', transform=train_transform)
val_ds.dataset = SteerDataSet(os.path.join(script_path, '..', 'data', 'train'), '.jpg', transform=valid_transform)

###################
## Train dataset ##
###################
print("The train dataset contains %d images " % len(train_ds))
# Extract all labels
labels = [train_ds[i][1] for i in range(len(train_ds))]
counts = [labels.count(i) for i in range(7)]
#data loader nicely batches images for the training process and shuffles (if desired)
weights = [1/c for c in counts]
sample_weights = [weights[lab] for lab in labels]
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=int(len(train_ds)*2.0), replacement=True)
trainloader = DataLoader(train_ds,batch_size=batch_size,sampler=sampler)
all_y = []
for S in trainloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
print(all_lbls, all_counts)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Training Dataset')
plt.show()

# visualise some images and print labels -- check these seem reasonable
example_ims, example_lbls = next(iter(trainloader))
print(' '.join(f'{example_lbls[j]}' for j in range(len(example_lbls))))
imshow(torchvision.utils.make_grid(example_ims))


########################
## Validation dataset ##
########################

print("The train dataset contains %d images " % len(val_ds))

#data loader nicely batches images for the training process and shuffles (if desired)
valloader = DataLoader(val_ds,batch_size=batch_size)
all_y = []
for S in valloader:
    im, y = S    
    all_y += y.tolist()

print(f'Input to network shape: {im.shape}')

#visualise the distribution of GT labels
all_lbls, all_counts = np.unique(all_y, return_counts = True)
plt.bar(all_lbls, all_counts, width = (all_lbls[1]-all_lbls[0])/2)
plt.xlabel('Labels')
plt.ylabel('Counts')
plt.xticks(all_lbls)
plt.title('Validation Dataset')
plt.show()

#######################################################################################################################################
####     INITIALISE OUR NETWORK                                                                                                    ####
#######################################################################################################################################

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
    

net = Net()

#######################################################################################################################################
####     INITIALISE OUR LOSS FUNCTION AND OPTIMISER                                                                                ####
#######################################################################################################################################

#for classification tasks
criterion = nn.CrossEntropyLoss()
#You could use also ADAM
optimizer = optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)


#######################################################################################################################################
####     TRAINING LOOP                                                                                                             ####
#######################################################################################################################################

# net.load_state_dict(torch.load('steer_net_last.pth', weights_only=True))
losses = {'train': [], 'val': []}
accs = {'train': [], 'val': []}
best_acc = 0
for epoch in range(400):  # loop over the dataset multiple times #epoc 14 (20 was good)
    net.train()
    epoch_loss = 0.0
    correct = 0
    total = 0
    if epoch == 200:
        optimizer = optim.Adam(net.parameters(), lr=0.0005, weight_decay=0.001)
    if epoch == 300:
        optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=0.001)
        
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        # print(loss, outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        epoch_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Epoch {epoch + 1} loss: {epoch_loss / len(trainloader)}')
    losses['train'] += [epoch_loss / len(trainloader)]
    accs['train'] += [100.*correct/total]
 
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in full_dataset.class_labels}
    total_pred = {classname: 0 for classname in full_dataset.class_labels}

    # again no gradients needed
    val_loss = 0
    with torch.no_grad():
        net.eval()
        for data in valloader:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            val_loss += loss.item()
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[full_dataset.class_labels[label.item()]] += 1
                total_pred[full_dataset.class_labels[label.item()]] += 1

    # print accuracy for each class
    class_accs = []
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        class_accs += [accuracy]

    val_acc = np.sum(list(correct_pred.values()))/np.sum(list(total_pred.values()))*100.0
    accs['val'] += [val_acc]
    losses['val'] += [val_loss/len(valloader)]

    if val_acc > best_acc:
        torch.save(net.state_dict(), 'steer_net_bestest.pth')
        best_acc = val_acc

    torch.save(net.state_dict(), 'steer_net_last.pth')

print('Finished Training')

plt.plot(losses['train'], label = 'Training')
plt.plot(losses['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(accs['train'], label = 'Training')
plt.plot(accs['val'], label = 'Validation')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

#######################################################################################################################################
####     PERFORMANCE EVALUATION                                                                                                    ####
#######################################################################################################################################
net.load_state_dict(torch.load('steer_net_bestest.pth', weights_only=True))

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    net.eval()
    for data in valloader:
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')

# prepare to count predictions for each class
correct_pred = {classname: 0 for classname in full_dataset.class_labels}
total_pred = {classname: 0 for classname in full_dataset.class_labels}

# again no gradients needed
actual = []
predicted = []
with torch.no_grad():
    for data in valloader:
        images, labels = data
        outputs = net(images)
        _, predictions = torch.max(outputs, 1)

        actual += labels.tolist()
        predicted += predictions.tolist()

        # collect the correct predictions for each class
        for label, prediction in zip(labels, predictions):
            if label == prediction:
                correct_pred[full_dataset.class_labels[label.item()]] += 1
            total_pred[full_dataset.class_labels[label.item()]] += 1

cm = metrics.confusion_matrix(actual, predicted, normalize = 'true')
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=full_dataset.class_labels)
disp.plot()
plt.show()

# print accuracy for each class
for classname, correct_count in correct_pred.items():
    accuracy = 100 * float(correct_count) / total_pred[classname]
    print(f'Accuracy for class: {classname:5s} is {accuracy:.1f}%')

