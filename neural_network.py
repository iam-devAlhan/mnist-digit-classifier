import numpy as np

class NeuralNetwork:
    
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W1 = np.random.randn(self.input_size, self.hidden_size) * 0.1
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size) * 0.1
        self.b2 = np.zeros((1, self.output_size))

    def relu(self, x):
        return np.maximum(0, x)
    
    def forward(self, X):
        self.Z1 = np.dot(self.W1, X) + self.b1

        self.A1 = self.relu(0, self.Z1)

        self.Z2 = np.dot(self.AI, self.W2) + self.b2

        exp_z = np.exp(self.Z2 - np.max(self.Z2, keepdims=True, axis=1))
        self.A2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return self.A2
    
