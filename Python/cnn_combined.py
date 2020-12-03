'''

In this file I test to combine two pre-trained models and OpenCV Canny Detector

'''

import torch
import torchvision
import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import time
import torchvision.models as models
import cv2

testDirectory = "/home/pinkie/Desktop/ANOII/Python/my_test_images/"

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Transform
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    torchvision.transforms.RandomVerticalFlip(),
    #torchvision.transforms.Resize(size)
    ])

### Net number 1
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
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
            nn.Linear(20 * 20 * 40, 200),
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


### Evaulate function
def evaulateOnTestFolder(num_images):
    # Load first net -> must be 80x80 for this net
    net = ConvNet()
    net.load_state_dict(torch.load('./models/cnnmixedForCombined80.pth'))

    # Load second net -> the second biggest net with 80x80 image input
    net2 = models.mobilenet_v2()
    net2.load_state_dict(torch.load('./models/cnn_mobileNet_Forcombined.pth'))

    ### Data loading
    classes = ('free', 'full')

    results = []
    with torch.no_grad():
        for i in range(1, num_images + 1):
            pytorch1 = 0
            pytorch2 = 0
            opencv = 0

            imForNet = Image.open(testDirectory + "test" + str(i) + ".jpg")
            imForNet = transform(imForNet).float()
            imForNet = imForNet.unsqueeze(0)

            # First model
            outputs = net(imForNet)
            _, predicted = torch.max(outputs, 1)

            if (classes[predicted[0]] == "full"):
                pytorch1 = 1
            else:
                pytorch1 = 0

            # Second model
            outputs = net2(imForNet)
            _, predicted = torch.max(outputs, 1)

            if (classes[predicted[0]] == "full"):
                pytorch2 = 1
            else:
                pytorch2 = 0

            # OpenCV non training
            opencv = cv2Process(i)

            # Evaluate - when 2 or more have same result
            res = 1 if (pytorch1 + pytorch2 + opencv >= 2) else 0
            results.append(str(res))

            # print every 100 images
            if(i % 100 == 0):
                #showImages(i-99,i,results)
                print(i)

    # Load Groundtruth file and compare with combined result
    grnd = loadGroundtruth()
    compare(results, grnd)

### OpenCV non training
def cv2Process(imNum):
    img = cv2.imread(testDirectory + "test" + str(imNum) + ".jpg", 0)
    median = cv2.medianBlur(img, 3)
    edges = cv2.Canny(median, 100, 100)
    # cv2.imshow('image',edges)
    # cv2.waitKey(0)

    threshold = 370
    pixelSum = 0
    for x in range(edges.shape[1]):
        for y in range(edges.shape[0]):
            val = edges[x, y]
            if (val > 150):
                pixelSum = pixelSum + 1
    return 1 if pixelSum > threshold else 0

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
    falseResults = []

    for i in range(0, len(ground)):
        detect = result[i]
        grnd = ground[i]
        tmpres.append((result[i], ground[i]))
        if (detect == "1" and grnd == "0"): 
            falsePositives = falsePositives + 1
            falseResults.append(i)
        if (detect == "0" and grnd == "1"): 
            falseNegatives = falseNegatives + 1
            falseResults.append(i)
        if (detect == "1" and grnd == "1"): truePositives = truePositives + 1
        if (detect == "0" and grnd == "0"): trueNegatives = trueNegatives + 1


    showFalseImages(falseResults, result)
    print(len(falseResults))

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
    

def showImages(bottom, top, results):
    imgarray = []
    for i in range(bottom, top):
        tmpimg = cv2.imread(testDirectory + "test" + str(i) + ".jpg")
        if (results[i - 1] == str(0)):
            cv2.circle(tmpimg, (36, 4), 6, (255, 0, 0), -1)
        elif (results[i - 1] == str(1)):
            cv2.circle(tmpimg, (36, 4), 6, (0, 255, 0), -1)
        imgarray.append(tmpimg)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(10, 10),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, imgarray):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()

def showFalseImages(falseImages, results):
    imgarray = []
    for i in falseImages:
        tmpimg = cv2.imread(testDirectory + "test" + str(i) + ".jpg")
        imgarray.append(tmpimg)

    fig = plt.figure(figsize=(4., 4.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                    nrows_ncols=(4, 10),  # creates 2x2 grid of axes
                    axes_pad=0.1,  # pad between axes in inch.
                    )

    for ax, im in zip(grid, imgarray):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)

    plt.show()


evaulateOnTestFolder(1344) #1344