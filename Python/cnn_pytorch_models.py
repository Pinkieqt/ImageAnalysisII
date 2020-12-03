'''

In this file I tried to train and evaulate pre-created models from PyTorch Models library (torchvision.models)

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
import torchvision.models as models
import time

### Device to use
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

### Hyper parameters
num_epochs = 10
batch_size = 128
learning_rate = 0.01
size = 80
PATH = './models/cnn_mobileNet_Forcombined.pth'

### Transform
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
    #torchvision.transforms.RandomVerticalFlip(),
    #torchvision.transforms.Resize(size)
    ])

### Train function
def train():
    ### Data loading
    directory="/home/pinkie/Desktop/ANOII/Python/train_images/"
    testDirectory = "/home/pinkie/Desktop/ANOII/Python/my_test_images/"

    trainds = torchvision.datasets.ImageFolder(root=directory, transform=transform)
    trainloader = torch.utils.data.DataLoader(
            trainds,
            batch_size=batch_size,
            num_workers=0,
            shuffle=True
    )

    classes = ("full", "free")

    # Models used and tested
    model = models.mobilenet_v2()
    #model = models.alexnet()
    #model = models.vgg11()
    #model = models.squeezenet1_0()
    #model = models.googlenet()
    #model = models.resnet18()
    model = model.to(device)

    # Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=0)
   
    # Track time
    start_time = time.time()
    n_total_steps = len(trainloader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):
            # origin shape: [32, 3, 32, 32] -> batch size, depth, x, y
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            #loss = criterion(outputs.logits, labels) # used for google net

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
    # Models used and tested
    net = models.mobilenet_v2()
    #net = models.alexnet()
    #net = models.vgg11()
    #net = models.squeezenet1_0()
    #net = models.googlenet()
    #net = models.mobilenet_v2()
    #net = models.resnet18()
    net.load_state_dict(torch.load(PATH))
    net.eval()

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

### Evaulate
evaulateOnTestFolder(1344)