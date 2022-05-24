import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt

from network import BuildModel

def main():
    # load images and labels
    print("Loading data...")
    with open('images.data', 'rb') as imagesf, open('labels.data', 'rb') as labelsf:
        images = pickle.load(imagesf)
        labels = pickle.load(labelsf)

    images = np.array(images, dtype=float) / 255
    images = np.reshape(images, (21700, 1, 256, 256))
    print("Done.")

    print("Starting Training...")
    model, train_acc, test_acc, train_loss, e_itr, bestAccuracy = BuildModel(images, labels)

    torch.save(model.state_dict(), "model.pt")
    
    # print final test accuracy and best accuracy
    print("Final Test Accuracy: %.2f%%"%(100*test_acc[-1]))
    print("Best Test Accuracy: %.2f%% (Epoch %d)"%(100*bestAccuracy[0],bestAccuracy[1]))

    # plot the test accuracy
    plt.plot(e_itr, test_acc)
    plt.title('Test Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # plot the training accuracy
    plt.plot(e_itr, train_acc)
    plt.title('Training Accuracy vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.show()

    # plot the training loss
    plt.plot(e_itr, train_loss)
    plt.title('Training Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

main()