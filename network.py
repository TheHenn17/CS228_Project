import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
import numpy as np
import math

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride):
    # calls pytorch conv2d layer, which can specify the output channels (therefore, # of 3x3 filters)
    # stride = 1 (because padding = 1) will keep the H x W dimensions the same, otherwise no
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

# Residual block
class BasicBlock(nn.Module):
    # This function is invoked when you create an instance of the nn.Module. 
    # Define the various parameters of a layer using self.param (to use in forward)
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride) # first conv; changes dimensions if stride != 1
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = conv3x3(out_channels, out_channels, 1) # second conv; keeps dimensions of first
        self.downsample = downsample # set downsample to the passed nn.module created in ResNet

    # Define how your output is computed. X is the input to the module, out is the output
    def forward(self, x):
        residual = x # set residual to input to skip layer if need be
        out = self.conv1(x) # first conv
        out = self.bn1(out) 
        out = self.relu(out)
        out = self.conv2(out) # second conv
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x) # downsample residual to match new dimensions
        out += residual # shortcut layer
        out = self.relu(out)
        return out

# ResNet
class ResNet20(nn.Module):
    def __init__(self):
        super(ResNet20, self).__init__()
        self.conv = conv3x3(3, 16, 1) # first conv layer; dimensions 32x32x16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, 16, 1) # first layer; dimensions 32x32x16, no downsampling
        self.layer2 = self.make_layer(16, 32, 2) # second layer; dimensions 16x16x32 (stride=2), downsampling
        self.layer3 = self.make_layer(32, 64, 2) # third layer; dimensions 8x8x64 (stride=2), downsampling
        self.avg_pool = nn.AvgPool2d(8) # flatten layer 8x8x64 into 1x1x64
        self.fc = nn.Linear(64, 10) # final linear layer classifies the 10 classes

    # Make a layer
    def make_layer(self, in_channels, out_channels, stride):
        downsample = None

        # downsamples the input to match the stride change and allow shortcut
        if (stride != 1) or (in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(in_channels, out_channels, stride), 
                nn.BatchNorm2d(out_channels))

        # creates the layer as a sequence of two Basic Blocks
        layer = nn.Sequential(
            BasicBlock(in_channels, out_channels, stride, downsample), # first block may downsample
            BasicBlock(out_channels, out_channels, 1, None)) # second block never downsamples

        return layer

    def forward(self, x):
        # x; 32x32x3
        out = self.conv(x) # first conv layer, 32x32x16
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out) # layer 1, 32x32x16
        out = self.layer2(out) # layer 2, downsamples 16x16x32
        out = self.layer3(out) # layer 3, downsamples, 8x8x64
        out = self.avg_pool(out) # flatten, 1x1x64
        out = out.view(out.size(0), -1) # ?
        out = self.fc(out) # fully connected layer classifies image
        return out

# performs mixup augmentation
def mixup(inputs, targets, alpha):
    size = targets.size(0) # size of the batch
    inputs2 = torch.zeros([inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)]) # initalize a new batch
    
    # one-hot encode classes so that mixup makes sense
    targets2 = torch.zeros([size,10]) 
    for i in range(size):
        targets2[i][targets[i]] = 1
    targets = targets2

    targets2 = torch.zeros([size,10]) # initalize new targets

    lam = np.random.beta(alpha, alpha) # get lambda value

    # get batch size new instances of mixups
    for i in range(size):
        samp = random.sample(range(size), 2) # two random instances from batch

        # perform the mixup, store in new batch
        inputs2[i] = (lam*inputs[samp[0]]) + ((1 - lam)*inputs[samp[1]])
        targets2[i] = (lam*targets[samp[0]]) + ((1 - lam)*targets[samp[1]])

    # switch back from onehot encoding by getting the max class argument from mixup
    targets = torch.zeros([size]).type(torch.LongTensor) 
    for i in range(size):
        targets[i] = torch.argmax(targets2[i])

    return inputs2, targets

# performs cutout augmentation
def cutout(inputs, targets, K):
    inputs2 = torch.clone(inputs) # copy batch
    low = math.floor(K/2) # computes dimensions of the cutout square
    high = math.ceil(K/2)

    # for each batch instance
    for i in range(targets.size(0)):
        r = random.randint(0, 1) # 50% chance
        if(r == 1):
            # find center pixel of cutout
            h = random.randint(0, inputs.size(2))
            w = random.randint(0, inputs.size(3))

            # for the square cutout range, set values to 0
            for j in range(h-low, h+high):
                for k in range(w-low, w+high):
                    # check bounds
                    if((j >= 0 and j < inputs.size(2)) and (k >= 0 and k < inputs.size(3))):
                        inputs2[i][0][j][k] = 0
                        inputs2[i][1][j][k] = 0
                        inputs2[i][2][j][k] = 0

    return inputs2, targets

# performs standard augmentation
def standardAug(inputs, targets, K):
    # create new batch set (targets are the same)
    inputs2 = torch.zeros([inputs.size(0), inputs.size(1), inputs.size(2), inputs.size(3)])

    # for each batch instance
    for i in range(targets.size(0)):
        # pad the image with K 0's on either side
        padded = torch.nn.functional.pad(input=inputs[i], pad=(K,K,K,K,0,0), mode='constant', value=0)

        k1 = random.randint(-K, K) # generate the random shifts
        k2 = random.randint(-K, K)
        inputs2[i] = padded[:,(K+k1):(inputs.size(2)+K+k1),(K-k2):(inputs.size(3)+K-k2)] # crop image at shift location

        r = random.randint(0, 1) # 50% chance
        if(r == 1): 
            inputs[i] = inputs2[i]

            # flip rows of pixels (horizntal flip)
            for j in range(int(inputs.size(2)/2)):
                inputs2[i][0][j] = inputs[i][0][inputs.size(2)-1-j]
                inputs2[i][0][inputs.size(2)-1-j] = inputs[i][0][j]
                inputs2[i][1][j] = inputs[i][1][inputs.size(2)-1-j]
                inputs2[i][1][inputs.size(2)-1-j] = inputs[i][1][j]
                inputs2[i][2][j] = inputs[i][2][inputs.size(2)-1-j]
                inputs2[i][2][inputs.size(2)-1-j] = inputs[i][2][j]

    return inputs2, targets

# builds model for a number of epochs, tests model every epoch
def BuildModel(X_train, Y_train, X_test, Y_test, epochs=100, learning_rate=0.001, batch_size=64, doMixup=False, doCutout=False, doStandard=False):
    model = ResNet20().to("cpu")

    tensor_x = torch.Tensor(X_train) # transform to torch tensor
    tensor_y = torch.Tensor(Y_train)
    tensor_y = tensor_y.type(torch.LongTensor) # long tensor needed for targets, not float

    dataset = TensorDataset(tensor_x,tensor_y) # create train datset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) # create train dataloader

    tensor_x = torch.Tensor(X_test) # transform to torch tensor
    tensor_y = torch.Tensor(Y_test)
    tensor_y = tensor_y.type(torch.LongTensor) # long tensor needed for targets, not float

    dataset = TensorDataset(tensor_x,tensor_y) # create test datset
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False) # create test dataloader

    criterion = nn.CrossEntropyLoss() # set loss model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # set optimizer

    # initalize return variables
    train_acc = []
    test_acc = []
    train_loss = []
    e_itr = []
    bestAccuracy = (0,-1)

    for epoch in range(1, epochs+1):
        model.train() # set to train mode after testing at the end of last epoch
        epoch_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            if(doStandard):
                inputs, targets = standardAug(inputs, targets, 4) # standard augmentation

            if(doCutout):
                inputs, targets = cutout(inputs, targets, 16) # cutout

            if(doMixup):
                inputs, targets = mixup(inputs, targets, 0.2) # mixup

            inputs = inputs.to("cpu")
            targets = targets.to("cpu")

            optimizer.zero_grad() # gradient calculation
            outputs = model(inputs) # generate outputs
            loss = criterion(outputs, targets) # calculate loss
            loss.backward() 
            optimizer.step() # make step based of gradient and learning rate

            epoch_loss += loss.item() # add up loss
            _, predicted = outputs.max(1) # get predicted class
            total += targets.size(0) # update total
            correct += predicted.eq(targets).sum().item() # get number of correct predicted labels

        train_acc.append(correct/total) # calculate and store training accuracy
        train_loss.append(epoch_loss) # store training loss
        e_itr.append(epoch) # store iteration number (for graph x-axis)

        model.eval() # switch to evaluation mode for testing
        correct = 0
        total = 0
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(test_loader):
                inputs = inputs.to("cpu")
                targets = targets.to("cpu")
                outputs = model(inputs) # generate outputs from model

                _, predicted = outputs.max(1) # get class prediction
                total += targets.size(0) # update total
                correct += predicted.eq(targets).sum().item() # get number of correct predicted labels
        
        # Calculate and store test accuracy, update best accuracy
        test_acc.append(correct/total)
        if(test_acc[-1] >= bestAccuracy[0]):
            bestAccuracy = (test_acc[-1],epoch)

        # Training marker
        print("Epoch[%d/%d] | Training Loss: %.4f | Training Acc: %.2f%% | Test Acc: %.2f%%"
              %(epoch, epochs, epoch_loss, 100*train_acc[-1], 100*test_acc[-1]))
    
    return train_acc, test_acc, train_loss, e_itr, bestAccuracy