import numpy as np
from keras.datasets import mnist


class NeuralNetwork:

    def __init__(self, input_size, hidden_size, output_size):
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # For Hidden Layer Activation Function Computation (1st layer)
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)

         # For Output Layer, Softmax applied
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        exp_z = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.A2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)

        return self.A2
    
    def predict_class(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)


       


def load_model(weights_file="mnist_weights.npz"):
    nn = NeuralNetwork(input_size=784, hidden_size=128, output_size=10)

    data = np.load(weights_file)

    nn.W1 = data["W1"]
    nn.b1 = data["b1"]
    nn.W2 = data["W2"]
    nn.b2 = data["b2"]

    # Return the loaded model with known weights and biases
    return nn

nn = load_model('mnist_weights.npz')

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test.reshape(10000, 784) / 255.0

# Make predictions
predictions = nn.predict_class(x_test)
accuracy = np.mean(predictions == y_test)

print(f"Test accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Show some predictions
print("\nSample predictions (first 20 test images):")
print("-" * 50)
for i in range(20):
    status = "✓" if predictions[i] == y_test[i] else "✗"
    print(f"Sample {i:2d}: Predicted={predictions[i]}, Actual={y_test[i]} {status}")

for i in range(5):
    single_image = x_test[i].reshape(1, 784)
    pred = nn.predict_class(single_image)
    confidence = nn.forward(single_image)[0][pred[0]]
    print(f"Image {i}: Predicted={pred[0]}, Actual={y_test[i]}, Confidence={confidence:.2%}")

print("\n✅ Model is working correctly!")