'''

In this file I tried LeNet5 and ConvolutionalNet found on PyTorch forums.

'''

import torch
import torchvision
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import time

### Device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Hyper parameters for training
num_epochs = 25
batch_size = 128
learning_rate = 0.01
size = 80
PATH = './models/cnn.pth'

### Transform data to tensor and resize/flip etc..
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    #torchvision.transforms.RandomVerticalFlip(),
    #torchvision.transforms.Resize(size)
    ])

### Convolutional Net
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Declare all the layers for feature extraction
        self.features = nn.Sequential(nn.Conv2d(in_channels=3,out_channels=5,kernel_size=3,stride=1,padding=1), 
                                      nn.ReLU(inplace=True),
                                      nn.Conv2d(in_channels=5,out_channels=10,kernel_size=3,stride=1,padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(10),
                                      nn.Conv2d(in_channels=10,out_channels=20,kernel_size=3,stride=1,padding=1),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(20),
                                      nn.Conv2d(in_channels=20,out_channels=40,kernel_size=3,stride=1,padding=1),
                                      nn.MaxPool2d(2, 2),
                                      nn.ReLU(inplace=True),
                                      nn.BatchNorm2d(40))
        
        self.classifier = nn.Sequential(
            nn.Linear(20 * 20 * 40, 200), # 10*10*40 for size of 40, 20 * 20 * 40 for size 80
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(200, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 2))
        
    def forward(self, x):
        x = self.features(x)
        #print(x.shape)
        x = x.view(-1, 20 * 20 * 40)
        x = self.classifier(x)
        return x

### LeNet
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2) # (80-5 + 0)/1 + 1
        self.conv2 = nn.Conv2d(6, 16, 5)
        # (W-F + 2P)/S + 1  16*17*17 for 80x80 img, 16*5*5 for 32 img
        self.fc1 = nn.Linear(16*17*17, 120)  
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.shape)
        x = x.view(-1, 16*17*17) #flatten 16*17*17 for 80x80 img, 16*5*5 for 32 img
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # dont need to call because its already in crossentropyloss fce
        return x



### Train function
def train():
    ### Data loading
    directory="/home/pinkie/Desktop/ANOII/Python/train_images/"

    trainds = torchvision.datasets.ImageFolder(root=directory, transform=transform)
    trainloader = torch.utils.data.DataLoader(
            trainds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
    )

    model = ConvNet().to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)

    # Track time
    start_time = time.time()
    n_total_steps = len(trainloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % n_total_steps == 0:
                print (f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.30f}')


    print('Finished Training')
    print("--- %s seconds ---" % (time.time() - start_time))
    torch.save(model.state_dict(), PATH)

### Evaulate function
def evaulateOnTestFolder(num_images):
    net = ConvNet()
    net.load_state_dict(torch.load(PATH))

    ### Data loading
    testDirectory = "/home/pinkie/Desktop/ANOII/Python/my_test_images/"
    classes = ('free', 'full')

    results = []
    with torch.no_grad():
        for i in range(1, num_images + 1):
            im = Image.open(testDirectory + "test" + str(i) + ".jpg")
            im = transform(im).float()
            im = im.unsqueeze(0)

            outputs = net(im)
            _, predicted = torch.max(outputs, 1)

            if (classes[predicted[0]] == "full"):
                results.append(str(1))
            else:
                results.append(str(0))

            if(i % 100 == 0):
                print(i)

    # Evaulate
    grnd = loadGroundtruth()
    compare(results, grnd)

### Load grountruth.txt
def loadGroundtruth():
    path = "/home/pinkie/Desktop/ANOII/Python/groundtruth.txt"
    lines = []
    with open(path) as f:
        lines = [line.rstrip() for line in f]
    
    return lines

### Compare groundtruth with result
def compare(result, ground):

    falsePositives = 0
    falseNegatives = 0
    truePositives = 0
    trueNegatives = 0

    tmpres = []

    for i in range(0, len(ground)):
        detect = result[i]
        grnd = ground[i]
        tmpres.append((result[i], ground[i]))
        if (detect == "1" and grnd == "0"): falsePositives = falsePositives + 1
        if (detect == "0" and grnd == "1"): falseNegatives = falseNegatives + 1
        if (detect == "1" and grnd == "1"): truePositives = truePositives + 1
        if (detect == "0" and grnd == "0"): trueNegatives = trueNegatives + 1


    print("falsePositives: " + str(falsePositives))
    print("falseNegatives: " + str(falseNegatives))
    print("truePositives: " + str(truePositives))
    print("trueNegatives: " + str(trueNegatives))

    acc = (truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives)
    precision = truePositives / (truePositives + falsePositives)
    sensitivity = truePositives / (truePositives + falseNegatives)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    print("F1: " + str(f1))
    print("Accuracy: " + str(acc))
    
### Train
#train()

###Evaulate
evaulateOnTestFolder(1344)