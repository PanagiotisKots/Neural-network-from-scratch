import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pickle
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# --- Activation Functions ---
def sigmoid(x):
    """Sigmoid activation function: f(x) = 1 / (1 + e^(-x))"""
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    """Derivative of the sigmoid function."""
    fx = sigmoid(x)
    return fx * (1 - fx)

def relu(x):
    """ReLU activation function: f(x) = max(0, x)"""
    return np.maximum(0, x)

def deriv_relu(x):
    """Derivative of ReLU: 1 for x > 0, otherwise 0"""
    return np.where(x > 0, 1, 0)

def mse_loss(y_true, y_pred):
    """Mean Squared Error loss function."""
    return ((y_true - y_pred) ** 2).mean()

# --- Neural Network Class ---
class OurAdvancedNeuralNetwork:
    """
    Advanced neural network with:
    - 2 inputs
    - 1 hidden layer with 3 neurons using ReLU activation
    - 1 output layer with 1 neuron using Sigmoid activation
    """

    def __init__(self):
        # Initialize weights and biases with random values
        self.w1 = np.random.normal(size=(2, 3))  # Layer 1: 2 inputs -> 3 hidden neurons
        self.b1 = np.random.normal(size=3)        # Hidden layer biases

        self.w2 = np.random.normal(size=3)        # Layer 2: 3 hidden neurons -> 1 output neuron
        self.b2 = np.random.normal()               # Output layer bias

        # Adam optimizer parameters
        self.m_w1, self.v_w1 = 0, 0  # For weights of layer 1
        self.m_w2, self.v_w2 = 0, 0  # For weights of layer 2
        self.m_b1, self.v_b1 = 0, 0  # For biases of layer 1
        self.m_b2, self.v_b2 = 0, 0  # For bias of output layer

        self.beta1, self.beta2 = 0.9, 0.999  # Adam parameters
        self.epsilon = 1e-8  # To prevent division by zero

    def feedforward(self, x):
        """Performs a forward pass through the network."""
        h = relu(np.dot(x, self.w1) + self.b1)  # Hidden layer output
        o = sigmoid(np.dot(h, self.w2) + self.b2)  # Output layer output
        return o

    def train(self, data, all_y_trues, epochs=1000, learn_rate=0.01, batch_size=2, lambda_reg=0.01):
        """Trains the network using mini-batch gradient descent and Adam optimizer."""
        losses = []  # Store loss for visualization

        for epoch in range(epochs):
            # Shuffle the data at the beginning of each epoch
            indices = np.random.permutation(len(data))
            data = data[indices]
            all_y_trues = all_y_trues[indices]

            # Process data in mini-batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                batch_y = all_y_trues[i:i + batch_size]

                # Initialize gradients for this batch
                d_w1, d_b1 = np.zeros_like(self.w1), np.zeros_like(self.b1)
                d_w2, d_b2 = np.zeros_like(self.w2), 0.0

                for x, y_true in zip(batch, batch_y):
                    # --- Forward pass ---
                    h = relu(np.dot(x, self.w1) + self.b1)  # Hidden layer output
                    o = sigmoid(np.dot(h, self.w2) + self.b2)  # Output layer output
                    y_pred = o

                    # --- Backward pass (Calculate gradients) ---
                    d_L_d_ypred = -2 * (y_true - y_pred)  # Loss gradient
                    d_ypred_d_w2 = h * deriv_sigmoid(np.dot(h, self.w2) + self.b2)
                    d_ypred_d_b2 = deriv_sigmoid(np.dot(h, self.w2) + self.b2)

                    # Hidden layer gradients
                    d_ypred_d_h = self.w2 * deriv_sigmoid(np.dot(h, self.w2) + self.b2)
                    d_h = d_ypred_d_h * deriv_relu(np.dot(x, self.w1) + self.b1)

                    # Accumulate gradients for the batch
                    d_w2 += d_L_d_ypred * d_ypred_d_w2
                    d_b2 += d_L_d_ypred * d_ypred_d_b2
                    d_w1 += np.outer(x, d_h)  # Correctly accumulates gradients for weights of the first layer
                    d_b1 += d_h  # d_h has the correct shape for bias

                # Apply Adam optimizer updates to weights and biases
                self.m_w1 = self.beta1 * self.m_w1 + (1 - self.beta1) * d_w1
                self.v_w1 = self.beta2 * self.v_w1 + (1 - self.beta2) * (d_w1 ** 2)
                self.w1 -= learn_rate * self.m_w1 / (np.sqrt(self.v_w1) + self.epsilon)

                self.m_b1 = self.beta1 * self.m_b1 + (1 - self.beta1) * d_b1
                self.v_b1 = self.beta2 * self.v_b1 + (1 - self.beta2) * (d_b1 ** 2)
                self.b1 -= learn_rate * self.m_b1 / (np.sqrt(self.v_b1) + self.epsilon)

                self.m_w2 = self.beta1 * self.m_w2 + (1 - self.beta1) * d_w2
                self.v_w2 = self.beta2 * self.v_w2 + (1 - self.beta2) * (d_w2 ** 2)
                self.w2 -= learn_rate * self.m_w2 / (np.sqrt(self.v_w2) + self.epsilon)

                self.m_b2 = self.beta1 * self.m_b2 + (1 - self.beta1) * d_b2
                self.v_b2 = self.beta2 * self.v_b2 + (1 - self.beta2) * (d_b2 ** 2)
                self.b2 -= learn_rate * self.m_b2 / (np.sqrt(self.v_b2) + self.epsilon)

            # Calculate loss at the end of each epoch
            y_preds = np.apply_along_axis(self.feedforward, 1, data)  # Predictions for the batch
            loss = mse_loss(all_y_trues, y_preds)  # Calculate loss
            losses.append(loss)  # Store loss for plotting

            # Print loss every 10 epochs
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.3f}")

        # Plot loss curve
        plt.plot(range(epochs), losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.show()

    def save_model(self, filename='model.pkl'):
        """Save the model to a file."""
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filename='model.pkl'):
        """Load a saved model from a file."""
        with open(filename, 'rb') as f:
            return pickle.load(f)

# --- Define Dataset ---
data = np.array([
    [-2, -1],  # Alice
    [25, 6],   # Bob
    [17, 4],   # Charlie
    [-15, -6], # Diana
])
all_y_trues = np.array([1, 0, 0, 1])  # Labels: 1 = Female, 0 = Male

# --- Train the Neural Network ---
network = OurAdvancedNeuralNetwork()
network.train(data, all_y_trues)

# --- Make Predictions ---
def classify_person(output):
    """Classifies a person as Male or Female based on the output probability."""
    confidence = output * 100  # Convert output to percentage
    return (f"{Fore.GREEN}Female ({confidence:.2f}% confidence)" 
            if output > 0.5 else f"{Fore.RED}Male ({100 - confidence:.2f}% confidence)")

# Define individuals to classify
individuals = [
    ("Alice", np.array([-2, -1])),
    ("Bob", np.array([25, 6])),
    ("Charlie", np.array([17, 4])),
    ("Diana", np.array([-15, -6]))
]

print("\n--- Predictions ---\n")
detailed_predictions = []  # Store detailed predictions for summary

for name, features in individuals:
    prediction = network.feedforward(features)  # Get the prediction from the network
    classification = classify_person(prediction)  # Classify the prediction
    
    # Store detailed result
    detailed_predictions.append((name, features, prediction, classification))

    # Display detailed result
    print(f"{Fore.CYAN}{name}:{Style.RESET_ALL}")
    print(f"  Input Features: Weight = {features[0]}, Height = {features[1]}")
    print(f"  Raw Output: {prediction:.4f}")
    print(f"  Classification: {classification}")
    print(f"  Prediction Threshold: 0.5\n")
    
# Summary of predictions
total_females = sum(1 for _, _, prediction, _ in detailed_predictions if prediction > 0.5)
total_males = len(individuals) - total_females

print(f"{Fore.MAGENTA}--- Summary ---{Style.RESET_ALL}")
print(f"Total Predictions Made: {len(individuals)}")
print(f"Predicted Females: {total_females}")
print(f"Predicted Males: {total_males}")

# Print detailed results
print(f"{Fore.MAGENTA}\n--- Detailed Predictions ---{Style.RESET_ALL}")
print(f"{'Name':<10} {'Weight':<10} {'Height':<10} {'Raw Output':<12} {'Classification'}")
print("-" * 60)
for name, features, prediction, classification in detailed_predictions:
    weight, height = features
    print(f"{name:<10} {weight:<10} {height:<10} {prediction:<12.4f} {classification}")


