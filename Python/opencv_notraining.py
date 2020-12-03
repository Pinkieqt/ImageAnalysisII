'''

In this file I test OpenCV Canny Detector

'''

import os
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
from PIL import Image
import cv2

testDirectory = "/home/pinkie/Desktop/ANOII/Python/my_test_images/"

### Evaulate function
def evaulateOnTestFolder(num_images):
    ### Data loading
    results = []
    for i in range(1, num_images + 1):
        opencv = 0

        # OpenCV non training
        opencv = cv2Process(i)

        results.append(str(opencv))

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
    
    threshold = 371
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

def showImagesRow(bottom, top, results):
    imgarray = []
    for i in range(bottom, top):
        tmpimg = cv2.imread(testDirectory + "test" + str(i) + ".jpg", 1)
        imgarray.append(tmpimg)


    numpy_horizontal = np.hstack(imgarray)

    numpy_horizontal_concat = np.concatenate(imgarray, axis=1)

    cv2.imshow('Numpy Horizontal Concat', numpy_horizontal_concat)
    cv2.waitKey()


evaulateOnTestFolder(1344) #1344