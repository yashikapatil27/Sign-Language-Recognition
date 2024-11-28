# Sign Language Recognition

This project implements various machine learning and deep learning models to classify hand gestures for sign language recognition using PyTorch. It includes logistic regression, artificial neural networks (ANNs), and multiple convolutional neural network (CNN) architectures, tested with different optimization strategies.

---

## Table of Contents

1. [Dataset](#Dataset)
2. [Models and Architectures](#models-and-architectures)
3. [Optimizers](#optimizers)
4. [Results](#results)

---

### Dataset
- The dataset consists of 64x64 grayscale images of hand gestures, stored in X.npy, with corresponding one-hot encoded labels in Y.npy.
- The images are reshaped into 64x64x1 tensors, while the labels are converted into integer classes.
- The dataset is split into training and testing sets using train_test_split with stratified sampling.

### Models and Architectures

1. Logistic Regression
- A simple linear model with one fully connected layer.

2. Artificial Neural Network (ANN)
 - A feedforward neural network with one hidden layer and ReLU activation.

3. Convolutional Neural Networks (CNNs)
- CNN_Model_01: Basic CNN with two convolutional layers and max-pooling.
- CNN_Model_02: Deeper CNN with additional dense layers for better classification.
- CNN_Model_03: Four convolutional layers and multiple dense layers for robust learning.
- CNN_Model_04: Six convolutional layers, deeper network for improved feature extraction.

### Optimizers
The following optimizers were tested for different models:

1. Stochastic Gradient Descent (SGD)
- Used with a learning rate of 0.01 and momentum of 0.9.

2. Adam
- Default learning rate of 0.001.
- Effective for handling sparse gradients and general convergence.

3. RMSprop
- Learning rate of 0.001.
- Used for stabilizing training in deeper models like CNN_Model_04.

4. Adagrad (experimentally tested on ANN)
- Adapts learning rates for each parameter, suitable for sparse data.

### Results

The following table summarizes the test accuracies of the implemented models:

| **Model**             | **Test Accuracy** | **Optimizer**   |
|-----------------------|-------------------|-----------------|
| Logistic Regression   | 58.4%             | SGD             |
| ANN                   | 55.0%             | Adam            |
| CNN_Model_01          | 84.2%             | Adam            |
| CNN_Model_02          | 92.3%             | RMSprop         |
| CNN_Model_03          | 96.7%             | Adam            |
| CNN_Model_04          | 97.8%             | RMSprop         |

Training metrics such as loss and accuracy are visualized for each epoch. Graphs are generated automatically.
