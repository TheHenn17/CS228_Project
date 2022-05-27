import math
import pickle
import numpy as np

from collections import Counter


# Images are just all of the images
# labels are the same length as images, 
# and just contain the author of the image
# at the corresponding index

with open('old_files/images.data', 'rb') as imagesf, open('old_files/labels.data', 'rb') as labelsf:
    images = pickle.load(imagesf)
    labels = pickle.load(labelsf)
        
images = np.array(images, dtype=object)
labels = np.array(labels, dtype=object)

# topAuthors is a list of top 100 most common labels
topAuthors = Counter(labels).most_common(10)

topAuthorIndicies = []

smallAut = 5000
for author in topAuthors:
    count = 0
    for i, val in enumerate(labels):
        if author[0] == val:
            if len(images[i]) <= 128 and len(images[i][0]) <= 128:
                count+=1
    if count < smallAut:
        smallAut = count
print(smallAut)
if smallAut >= 220:
    smallAut = 220
for author in topAuthors:
    candidates = []
    for i, val in enumerate(labels):
        if author[0] == val: # author found in labels
            if len(images[i]) <= 128 and len(images[i][0]) <= 128: # check whether image at this location is under 256x256
                candidates.append((i, (len(images[i])*len(images[i][0])))) # save location and total pixels
    candidates.sort(key = lambda x: x[1], reverse=True) # sort candidates by largest sizes
    for i in range(smallAut): 
        topAuthorIndicies.append(candidates[i][0]) # take 217 largest images

# TopAuthorIndecies is a list of all of the locations where there is a top 100 author's word
# that is also fitting the size constraint of 256x256

# topImages is a list of all the valid images
# topLabels is a corresponding list of all the labels of those valid images
topImages = []
topLabels = []
for index in topAuthorIndicies:
    topImages.append(images[index])
    topLabels.append(labels[index])

topImagesUnpadded = np.array(topImages, dtype=object)
topLabels = np.array(topLabels, dtype=object)

uniqueLab = np.unique(topLabels)
print(uniqueLab)
for i in range(topLabels.shape[0]):
    for j in range(uniqueLab.shape[0]):
        if topLabels[i] == uniqueLab[j]:
            topLabels[i] = j
print(topLabels[-1])

topImages = np.ones((topImagesUnpadded.shape[0],1,128,128))
for i in range(topImagesUnpadded.shape[0]):
    height = topImagesUnpadded[i].shape[0]
    width = topImagesUnpadded[i].shape[1]

    left = 64 - math.floor(width/2)
    right = 64 - math.ceil(width/2)
    top = 64 - math.floor(height/2)
    bottom = 64 - math.ceil(height/2)

    paddedImage = np.pad(array=topImagesUnpadded[i], pad_width=[(top,bottom),(left,right)], mode='constant', constant_values=1)
    topImages[i][0] = paddedImage

with open('images10.data', 'wb') as f:
    pickle.dump(topImages, f)

with open('labels10.data', 'wb') as f:
    pickle.dump(topLabels, f)
exit(0)

    #topImages = np.array(topImages, dtype=object)
    # Our data is set into topImages and topLabels

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(torch.cuda.get_device_name())

# # model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
# # model.fc = nn.Linear(in_features=512, out_features=10, bias=True)

# # model = model.to(device)

# X_train, x_test, y_train, y_test = train_test_split(topImages, topLabels, test_size=0.2)

# inputs = torch.from_numpy(X_train)
# labels = torch.from_numpy(y_train)

# dataset = torch.utils.data.TensorDataset(inputs, labels)
# trainloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# inputs = torch.from_numpy(x_test)
# labels = torch.from_numpy(y_test)

# dataset = torch.utils.data.TensorDataset(inputs, labels)
# testloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# quit()

# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# loss_history = []
# training_accuracy = []

# for epoch in range(100):  # loop over the dataset multiple times
#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = model(inputs.float())
#         loss = criterion(outputs, labels.flatten().long())
#         loss_history.append(loss.item())
#         loss.backward()
#         optimizer.step()

#     if (epoch + 1) % 20 == 0:
#         print('epoch {}, loss {}'.format(epoch + 1, loss.item()))