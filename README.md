# CS228_Project: Handwriting analysis using Deep Neural Networks

## The Problem and The Solution
For this project, we wanted to design a deep neural network that could match handwriting samples to a known author.
This type of network has all sorts of practical applications, such as identifying the handwriting on criminal documents.
We use the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) to train our network.
Our network makes use of non-standard convolusion filters that are rectangular in nature, the idea being that english handwriting is often characterized by broad strokes, either vertical or horizontal.
Please read the report PDF file in the repository for more details.

## The Network
- 16 Convolutional Layers
- 1 Pooling Layer and 1 Fully Connected layer
- Training Epochs: 100
- Learning rate: 0.001
- Activation Function: ReLU
- GoogleNet Style Inception Blocks
- Downsampling every two inception blocks
The images must be given in 192x192x1, where the color channel is greyscale. The handwritten text can be white on black or black on white

## How to Use
- Download the [IAM Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
- Run the dataPickler.py script
- Next, run the cleaner.py script to create the dataset and corresponding labels of the 100 authors with the most writing samples
- Finally, run main.py to train and test the network

## Results
- We are able to perform better than random guessing, though not by much
- However, since authors have have a similar style of handwriting, we decided to check the top 5 authors the nework predicted the test data could belong to
- About 1/3 of the time, the network could predict the correct author among it's top-5 predictions (out of 100 authors)
- Clearly this result is not desirable, but it does show that this type of network may be able to solve this problem, given more time and research
