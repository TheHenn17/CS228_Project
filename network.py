import torch
import torchvision
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# Inception block
class InceptionBlock(nn.Module):
    # out_channels should match in_channels by the end
    def __init__(self, in_channels):
        super(InceptionBlock, self).__init__()
        self.dim = int(in_channels/4) # reduce the dimension by 4, so all branches add up to same dimension
        self.conv53 = nn.Conv2d(in_channels, self.dim, kernel_size=(5,3), stride=1, padding=(2,1))
        self.conv73 = nn.Conv2d(in_channels, self.dim, kernel_size=(7,3), stride=1, padding=(3,1))
        self.conv35 = nn.Conv2d(in_channels, self.dim, kernel_size=(3,5), stride=1, padding=(1,2))
        self.conv37 = nn.Conv2d(in_channels, self.dim, kernel_size=(3,7), stride=1, padding=(1,3))
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
        out = torch.cat(out, 1)
        out = out + x
        out = self.relu(out)
        return out

# ResNet
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=1, stride=1, padding=0) # 192x192x1 --> 192x192x16
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(16, 32) # first inception layer; dimensions 96x96x32
        self.layer2 = self.make_layer(32, 64) # second inception layer; dimensions 48x48x64
        self.layer3 = self.make_layer(64, 128) # third inception layer; dimensions 24x24x128
        self.layer4 = self.make_layer(128, 256) # fourth inception layer; dimensions 12x12x256
        self.layer5 = self.make_layer(256, 512) # fifth inception layer; dimensions 6x6x512
        self.avg_pool = nn.AvgPool2d(6) # flatten layer 6x6x256 into 1x1x512
        self.fc = nn.Linear(512, 100) # final linear layer classifies the 100 authors

    # Make a layer
    def make_layer(self, in_channels, out_channels):
        layer = nn.Sequential(
            InceptionBlock(in_channels),
            InceptionBlock(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        return layer

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        out = self.relu(out)
        out = self.layer4(out)
        out = self.relu(out)
        out = self.layer5(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# builds model for a number of epochs, tests model every epoch
def BuildModel(images, labels, batch_size=64, learning_rate=0.001, epochs=100):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Selected Device:",device)
    model = Network().to(device)

    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=(float(1/6)), stratify=labels)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(size=192, pad_if_needed=True),
        torchvision.transforms.RandomAffine(degrees=45, translate=(0.5,0.5), scale=(0.5,2)),
        torchvision.transforms.RandomHorizontalFlip(p=0.5),
        torchvision.transforms.RandomVerticalFlip(p=0.5),
        torchvision.transforms.RandomPerspective(distortion_scale=0.1, p=0.5)
    ])

    tensor_x = torch.Tensor(X_train) # transform to torch tensor
    tensor_x = torchvision.transforms.functional.invert(tensor_x)
    tensor_y = torch.Tensor(Y_train)
    tensor_y = tensor_y.type(torch.LongTensor) # long tensor needed for targets, not float

    dataset = TensorDataset(tensor_x,tensor_y) # create train datset
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) # create train dataloader

    tensor_x = torch.Tensor(X_test) # transform to torch tensor
    tensor_x = torchvision.transforms.functional.invert(tensor_x)
    tensor_y = torch.Tensor(Y_test)
    tensor_y = tensor_y.type(torch.LongTensor) # long tensor needed for targets, not float

    dataset = TensorDataset(tensor_x,tensor_y) # create test datset
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True) # create test dataloader

    criterion = nn.CrossEntropyLoss() # set loss model
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # set optimizer

    # initalize return variables
    train_acc = []
    test_acc = []
    train_loss = []
    e_itr = []
    bestAccuracy = (0,-1)

    print("Entering Training Loop.")
    for epoch in range(1, epochs+1):
        model.train() # set to train mode after testing at the end of last epoch
        epoch_loss = 0
        correct = 0
        total = 0
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = transforms(inputs)
            inputs = inputs.to(device)
            targets = targets.to(device)

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
                inputs = inputs.to(device)
                targets = targets.to(device)

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
    
    return model, train_acc, test_acc, train_loss, e_itr, bestAccuracy