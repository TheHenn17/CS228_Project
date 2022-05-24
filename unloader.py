from cProfile import label
from unittest import TestLoader
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import math
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from sklearn.model_selection import train_test_split

import collections
from collections import Counter


# Images are just all of the images
# labels are the same length as images, 
# and just contain the author of the image
# at the corresponding index

with open('../images.data', 'rb') as imagesf, open('labels.data', 'rb') as labelsf:
    images = pickle.load(imagesf)
    labels = pickle.load(labelsf)
        
    images = np.array(images, dtype=object)
    labels = np.array(labels, dtype=object)

    # topAuthors is a list of top 100 most common labels
    topAuthors = Counter(labels).most_common(100)

    topAuthorIndecies = []

    for author in topAuthors:
        counter = 0
        for i, val in enumerate(labels):
            # if we find the author in the the list of all of the labels
            # We need to collect which index it is at
            if author[0] == val:
                # Before we add the index to the list of all locations where there is a top 100 author work,
                # We check whether is is less than 256x256
                if len(images[i]) <= 256 and len(images[i][0]) <= 256:
                    # For all purposes, we have found a perfect example, now to limit it to only 217 values total
                    counter = counter + 1
                    if counter < 218:
                        # If we are within out limit, we can append the index
                        topAuthorIndecies.append(i)
                    else:
                        break

    # TopAuthorIndecies is a list of all of the locations where there is a top 100 author's word
    # that is also fitting the size constraint of 256x256

    # topImages is a list of all the valid images
    # topLabels is a corresponding list of all the labels of those valid images
    topImages = []
    topLabels = []
    for index in topAuthorIndecies:
        topImages.append(images[index])
        topLabels.append(labels[index])

    topImagesUnpadded = np.array(topImages, dtype=object)
    topLabels = np.array(topLabels, dtype=object)

    # sizePerAuthor is the number of words for the least popular of the 100 authors ~217
    sizePerAuthor = Counter(topLabels).most_common(100)[-1][1]

    # topImages is now a nparray of all of the needed images, we need to pad them all to make them 256x256
    #52378

    topImages = []
    for image in topImagesUnpadded:
        height = image.shape[0]
        width = image.shape[1]

        left = 128 - math.floor(width/2)
        right = 128 - math.ceil(width/2)
        top = 128 - math.floor(height/2)
        bottom = 128 - math.ceil(height/2)

        paddedImage = torch.nn.functional.pad(input=torch.tensor(image), pad=(left,right,top,bottom), mode='constant', value=1)
        topImages.append(paddedImage)

    topImages = np.array(topImages, dtype=object)
    # Our data is set into topImages and topLabels

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.get_device_name())

    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
    model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

    model = model.to(device)

    X_train, x_test, y_train, y_test = train_test_split(topImages, topLabels, test_size=0.2)

    inputs = Variable(torch.from_numpy(X_train))
    labels = Variable(torch.from_numpy(y_train))

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    inputs = Variable(torch.from_numpy(x_test))
    labels = Variable(torch.from_numpy(y_test))

    dataset = torch.utils.data.TensorDataset(inputs, labels)
    testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    loss_history = []
    training_accuracy = []

    for epoch in range(100):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs.float())
            loss = criterion(outputs, labels.flatten().long())
            loss_history.append(loss.item())
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:
            print('epoch {}, loss {}'.format(epoch + 1, loss.item()))