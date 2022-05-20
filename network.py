import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# Inception block
class InceptionBlock(nn.Module):
    # out_channels should match in_channels by the end
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.dim = int(in_channels/4) # reduce the dimension by 4, so all branches add up to same dimension
        self.conv53 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, kernel_size=1, stride=1, padding=0), # downsample, then convolve
            nn.Conv2d(self.dim, self.dim, kernel_size=(5,3), stride=1, padding=(2,1)))
        self.conv73 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, kernel_size=1, stride=1, padding=0), # downsample, then convolve
            nn.Conv2d(self.dim, self.dim, kernel_size=(7,3), stride=1, padding=(3,1)))
        self.conv35 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, kernel_size=1, stride=1, padding=0), # downsample, then convolve
            nn.Conv2d(self.dim, self.dim, kernel_size=(3,5), stride=1, padding=(1,2)))
        self.conv37 = nn.Sequential(
            nn.Conv2d(in_channels, self.dim, kernel_size=1, stride=1, padding=0), # downsample, then convolve
            nn.Conv2d(self.dim, self.dim, kernel_size=(3,7), stride=1, padding=(1,3)))
        self.bn = nn.BatchNorm2d(self.dim)
        self.relu = nn.ReLU(inplace=True)

    # Define how your output is computed. X is the input to the module, out is the output
    def forward(self, x):
        # First branch
        out1 = self.conv53(x) 
        out1 = self.bn(out1) 
        out1 = self.relu(out1)

        # Second branch
        out2 = self.conv73(x) 
        out2 = self.bn(out2) 
        out2 = self.relu(out2)

        # Third branch
        out3 = self.conv35(x) 
        out3 = self.bn(out3) 
        out3 = self.relu(out3)

        # Fourth branch
        out4 = self.conv37(x) 
        out4 = self.bn(out4) 
        out4 = self.relu(out4)

        # Calculate out
        out = [out1, out2, out3, out4]
        out += x
        out = self.relu(out)
        return out

# ResNet
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv =  nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1) # first conv layer; 256x256x8
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(8, 16) # first layer; dimensions 128x128x16
        self.layer2 = self.make_layer(16, 32) # second layer; dimensions 64x64x32
        self.layer3 = self.make_layer(32, 64) # second layer; dimensions 32x32x64
        self.layer4 = self.make_layer(64, 128) # second layer; dimensions 16x16x128
        self.layer5 = self.make_layer(128, 256) # third layer; dimensions 8x8x256
        self.avg_pool = nn.AvgPool2d(8) # flatten layer 8x8x256 into 1x1x256
        self.fc = nn.Linear(256, 100) # final linear layer classifies the 100 authors

    # Make a layer
    def make_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            InceptionBlock(in_channels),
            InceptionBlock(in_channels),
            nn.Conv2d(in_channels, out_channels, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        return layer

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# builds model for a number of epochs, tests model every epoch
def BuildModel(X_train, Y_train, X_test, Y_test, epochs=100, learning_rate=0.001, batch_size=64, doMixup=False, doCutout=False, doStandard=False):
    model = Network().to("cpu")

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