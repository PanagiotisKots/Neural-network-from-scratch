# Simple Neural Network with Python and NumPy

This repository contains a simple neural network implementation from scratch using Python and NumPy. The network is designed to classify data based on two input features. While this code is meant for educational purposes, it demonstrates fundamental concepts in neural networks, including feedforward propagation, activation functions, loss calculation, and basic backpropagation.

## Code Structure

### Functions

- **sigmoid(x)**:  
  The sigmoid activation function, defined as:
  \[
  f(x) = \frac{1}{1 + e^{-x}}
  \]
  It compresses input values into a range between 0 and 1, making it suitable for binary classification problems.

- **deriv_sigmoid(x)**:  
  Calculates the derivative of the sigmoid function, used in backpropagation for updating weights.

- **mse_loss(y_true, y_pred)**:  
  Computes the Mean Squared Error (MSE) loss between true values (`y_true`) and predicted values (`y_pred`). This loss function evaluates the network's performance, with lower values indicating better predictions.

### Class: `OurNeuralNetwork`

The `OurNeuralNetwork` class defines a simple neural network with:
- **2 inputs** (representing two features in the dataset)
- **1 hidden layer** with **2 neurons** (`h1` and `h2`)
- **1 output layer** with **1 neuron** (`o1`)

#### Methods

- **__init__()**:  
  Initializes random weights and biases for each neuron.

- **feedforward(x)**:  
  Performs feedforward propagation to generate a prediction based on input features `x`. It calculates outputs by passing the weighted sums through the sigmoid function at each layer.

- **train(data, all_y_trues)**:  
  Trains the neural network on a dataset (`data`) with corresponding true labels (`all_y_trues`). It performs backpropagation to adjust weights and biases based on the calculated gradients, using Mean Squared Error (MSE) loss as a performance metric.

### Training the Network

The network is trained on a small dataset with two input features. During each epoch, it calculates the loss and adjusts weights and biases using backpropagation. After training, the network should be able to classify new data.

```python
# Define dataset
data = np.array([
  [-2, -1],  # Alice
  [25, 6],   # Bob
  [17, 4],   # Charlie
  [-15, -6], # Diana
])
all_y_trues = np.array([1, 0, 0, 1])  # Target labels for each input

# Train the network
network = OurNeuralNetwork()
network.train(data, all_y_trues)
