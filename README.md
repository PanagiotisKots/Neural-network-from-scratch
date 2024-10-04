OurNeuralNetwork: A Simple Neural Network in Python

This repository contains a basic implementation of a neural network from scratch using Python and NumPy. The code demonstrates fundamental concepts of neural networks, including forward propagation, backpropagation, and gradient descent, in an easily understandable way.

This network has been designed for educational purposes and is not optimized for production use. Its main goal is to help beginners grasp how a simple neural network works, including training and making predictions.
Table of Contents

    Overview
    Sigmoid Function
    Mean Squared Error Loss
    Neural Network Architecture
    Forward Propagation
    Backpropagation and Training
    Training the Model
    Making Predictions
    Usage
    How it Works
    Disclaimer
    License

Overview

This neural network consists of:

    2 input features
    1 hidden layer with 2 neurons (h1 and h2)
    1 output neuron (o1)

It is a fully connected feedforward neural network that can be trained using gradient descent with backpropagation. This network is suitable for binary classification tasks.

The code includes:

    Forward propagation to compute predictions
    Backpropagation to update the weights and biases using the gradients of the loss function
    Mean Squared Error (MSE) loss for the training process

Sigmoid Function

The sigmoid function is used as the activation function for both hidden and output layers:

python

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

The sigmoid function produces an output between 0 and 1, making it suitable for binary classification.

The derivative of the sigmoid function, used in backpropagation, is:

python

def deriv_sigmoid(x):
    fx = sigmoid(x)
    return fx * (1 - fx)

Mean Squared Error Loss

The mean squared error (MSE) is used as the loss function to measure how well the predictions match the true labels:

python

def mse_loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

During training, the network seeks to minimize this loss.
Neural Network Architecture

The class OurNeuralNetwork defines the structure of the network. It includes the initialization of weights and biases, forward propagation, and the training process.

python

class OurNeuralNetwork:
    def __init__(self):
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()

        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    Weights: w1, w2, ..., w6
    Biases: b1, b2, b3

These parameters are randomly initialized using a normal distribution.
Forward Propagation

The feedforward function calculates the output of the network based on the inputs and current weights and biases:

python

def feedforward(self, x):
    h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
    h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
    o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
    return o1

Backpropagation and Training

The network is trained using gradient descent, where the weights and biases are updated based on the derivative of the loss function with respect to each parameter. This is performed over multiple epochs, where in each epoch, the network:

    Feeds data forward to make predictions
    Computes the loss between the predictions and true values
    Updates weights and biases to reduce the loss using the chain rule for derivatives

The train method is responsible for this process:

python

def train(self, data, all_y_trues):
    learn_rate = 0.1
    epochs = 1000  # Number of iterations
    
    for epoch in range(epochs):
        for x, y_true in zip(data, all_y_trues):
            # Forward pass
            # Backpropagation
            # Update weights and biases

Training the Model

A sample dataset is provided to train the network. The dataset consists of four data points:

    Alice: [-2, -1] (labeled as 1)
    Bob: [25, 6] (labeled as 0)
    Charlie: [17, 4] (labeled as 0)
    Diana: [-15, -6] (labeled as 1)

python

data = np.array([[-2, -1], [25, 6], [17, 4], [-15, -6]])
all_y_trues = np.array([1, 0, 0, 1])

network = OurNeuralNetwork()
network.train(data, all_y_trues)

During training, the loss is printed every 10 epochs to monitor progress.
Making Predictions

After training, the network can make predictions for new individuals. A sample function classify_person is used to determine the class (either Female or Male) based on the output of the neural network:

python

def classify_person(output):
    if output > 0.5:
        return "Female"
    else:
        return "Male"

You can test the network with new data points:

python

individuals = [
    ("Emily", np.array([-7, -3])),
    ("Frank", np.array([20, 2])),
    ("Grace", np.array([10, -1])),
    # Add more individuals
]
for name, features in individuals:
    prediction = network.feedforward(features)
    print(f"{name}: {prediction:.3f} - {classify_person(prediction)}")

Usage

    Clone the repository:

    bash

git clone https://github.com/your-repo/neural-network-from-scratch.git
cd neural-network-from-scratch

Install NumPy:

bash

pip install numpy

Run the Python script:

bash

    python neural_network.py

    View the predictions and training progress in the console.

How it Works

    Initialization: Weights and biases are initialized randomly.
    Forward Pass: Input features are passed through the network to calculate predictions.
    Loss Calculation: The loss function (MSE) compares predictions to true values.
    Backpropagation: Gradients of the loss with respect to each weight and bias are calculated.
    Gradient Descent: Weights and biases are updated to reduce the loss.
    Prediction: After training, the network can be used to classify new data points.

Disclaimer

This code is intended as a simple educational example to explain basic neural network concepts. It is not optimized for real-world use and lacks features such as regularization, advanced optimizers, and proper validation techniques. For production, consider using libraries like TensorFlow, PyTorch, or Keras.
License

This project is licensed under the MIT License - see the LICENSE file for details.
