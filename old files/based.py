
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np

images = []
labels = []
for filename in os.listdir("xml"):
    f = os.path.join("xml", filename)

    tree = ET.parse(f)
    root = tree.getroot()
    writer = int(root.attrib['writer-id'])

    direct = os.path.join("words", filename[:3], filename[:-4])
    for fname in os.listdir(direct):
        f2 = os.path.join(direct, fname)

        print(f2)
        try:
            pic = plt.imread(f2)
            images.append(pic)
            labels.append(writer)
        except:
            continue

with open('images.data', 'wb') as f:
    pickle.dump(images, f)

with open('labels.data', 'wb') as f:
    pickle.dump(labels, f)

# with open('images.data', 'rb') as f:
#     data = pickle.load(f)