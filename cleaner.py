import math
import pickle
import numpy as np

from collections import Counter


# Images are just all of the images
# labels are the same length as images, 
# and just contain the author of the image
# at the corresponding index

with open('images.data', 'rb') as imagesf, open('labels.data', 'rb') as labelsf:
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
            if len(images[i]) <= 192 and len(images[i][0]) <= 192:
                count+=1
    if count < smallAut:
        smallAut = count
print(smallAut)
if smallAut >= 120:
    smallAut = 120
for author in topAuthors:
    candidates = []
    for i, val in enumerate(labels):
        if author[0] == val: # author found in labels
            if len(images[i]) <= 192 and len(images[i][0]) <= 192: # check whether image at this location is under 256x256
                candidates.append((i, (len(images[i])*len(images[i][0])))) # save location and total pixels
    candidates.sort(key = lambda x: x[1], reverse=True) # sort candidates by largest sizes
    for i in range(smallAut): 
        topAuthorIndicies.append(candidates[i][0]) 

# TopAuthorIndecies is a list of all of the locations where there is a top 100 author's word
# that is also fitting the size constraint of 192x192

# topImages is a list of all the valid images
# topLabels is a corresponding list of all the labels of those valid images
topImages = []
topLabels = []
for index in topAuthorIndicies:
    topImages.append(images[index])
    topLabels.append(labels[index])

topImagesUnpadded = np.array(topImages, dtype=object)
topLabels = np.array(topLabels, dtype=object)

# rename authors from 1-100
uniqueLab = np.unique(topLabels)
print(uniqueLab)
for i in range(topLabels.shape[0]):
    for j in range(uniqueLab.shape[0]):
        if topLabels[i] == uniqueLab[j]:
            topLabels[i] = j
print(topLabels[-1])

# pad all images to be 192x192
topImages = np.ones((topImagesUnpadded.shape[0],1,192,192))
for i in range(topImagesUnpadded.shape[0]):
    height = topImagesUnpadded[i].shape[0]
    width = topImagesUnpadded[i].shape[1]

    left = 96 - math.floor(width/2)
    right = 96 - math.ceil(width/2)
    top = 96 - math.floor(height/2)
    bottom = 96 - math.ceil(height/2)

    paddedImage = np.pad(array=topImagesUnpadded[i], pad_width=[(top,bottom),(left,right)], mode='constant', constant_values=1)
    topImages[i][0] = paddedImage

# save the new dataset
with open('images100.data', 'wb') as f:
    pickle.dump(topImages, f)

with open('labels100.data', 'wb') as f:
    pickle.dump(topLabels, f)