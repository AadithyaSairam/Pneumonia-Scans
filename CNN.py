import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
import torch.nn.functional as F 
from torch import nn,optim
from torchvision import transforms as T,datasets,models
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
from collections import OrderedDict
from tqdm import tqdm
pd.options.plotting.backend = "plotly"
from torch import nn, optim
from torch.autograd import Variable
from torchsummary import summary

def data_transforms(phase = None):
    
    if phase == TRAIN:

        data_T = T.Compose([
            
                T.Resize(size = (256,256)),
                T.RandomRotation(degrees = (-20,+20)),
                T.CenterCrop(size=224),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    
    elif phase == TEST or phase == VAL:

        data_T = T.Compose([

                T.Resize(size = (224,224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
        
    return data_T

data_dir = "C:/Users/aadis/Documents/PROJECTS/Pneumonia Scans/chest_xray/chest_xray"
TEST = 'test'
TRAIN = 'train'
VAL ='val'

trainset = datasets.ImageFolder(os.path.join(data_dir, TRAIN),transform = data_transforms(TRAIN))
testset = datasets.ImageFolder(os.path.join(data_dir, TEST),transform = data_transforms(TEST))
validset = datasets.ImageFolder(os.path.join(data_dir, VAL),transform = data_transforms(VAL))

class_names = trainset.classes
print(class_names)
print(trainset.class_to_idx)

trainloader = DataLoader(trainset,batch_size = 64,shuffle = True)
validloader = DataLoader(validset,batch_size = 64,shuffle = True)
testloader = DataLoader(testset,batch_size = 64,shuffle = True)

images, labels = next(iter(trainloader))
print(images.shape)
print(labels.shape)

for i, (images,labels) in enumerate(trainloader):
        if torch.cuda.is_available():
            images=Variable(images.cuda())
            labels=Variable(labels.cuda())

images.shape, labels.shape

class classify(nn.Module):
    def __init__(self):
        super(classify, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.dropout1 = nn.Dropout(0.1)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(256 * 9 * 9, 128)
        self.dropout4 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)
    
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.pool(x)
    
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)
    
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.pool(x)
    
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.pool(x)
    
        # Calculate the size of the tensor after the last max pooling operation
        x_size = x.size(1) * x.size(2) * x.size(3)
    
        x = x.view(-1, x_size)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
    
        return x

    
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

summary(classify(), (images.shape[1], images.shape[2], images.shape[3]))

model = classify()
# defining the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# defining the loss function
criterion = nn.CrossEntropyLoss()
# checking if GPU is available
if torch.cuda.is_available():
    model = model.cuda()
    criterion = criterion.cuda()

Losses = []
for i in range(10):
    running_loss = 0
    for images, labels in trainloader:
        
        #Changing images to cuda for gpu
        if torch.cuda.is_available():
          images = images.cuda()
          labels = labels.cuda()

        # Training pass
        # Sets the gradient to zero
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        # accumulates the loss for mini batch
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        Losses.append(loss)
        
        running_loss += loss.item()
    else:
        print("Epoch {} - Training loss: {}".format(i+1, running_loss/len(trainloader)))

correct_count, all_count = 0, 0
for images,labels in testloader:
  for i in range(len(labels)):
    if torch.cuda.is_available():
        images = images.cuda()
        labels = labels.cuda()
    img = images[i].view(1, 3, 224, 224)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.cpu()[0])
    pred_label = probab.index(max(probab))
    true_label = labels.cpu()[i]
    if(true_label == pred_label):
      correct_count += 1
    all_count += 1

print("Number Of Images Tested =", all_count)
print("\nModel Accuracy =", (correct_count/all_count))