import numpy as np
from keras.datasets import mnist

# ============================================
# YOUR NEURAL NETWORK CLASS (Same as before)
# ============================================
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Initialize weights and biases
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    
    def forward(self, X):
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.maximum(0, self.Z1)  # ReLU
        
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        
        # Softmax
        exp_z = np.exp(self.Z2 - np.max(self.Z2, axis=1, keepdims=True))
        self.A2 = exp_z / np.sum(exp_z, axis=1, keepdims=True)
        
        return self.A2
    
    def cross_entropy_loss(self, y_pred, y_true):
        epsilon = 1e-8
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        batch_size = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred)) / batch_size
        return loss
    
    def backward(self, X, y_true, learning_rate=0.1):
        batch_size = X.shape[0]
        
        # Output layer gradient
        dZ2 = self.A2 - y_true
        dW2 = np.dot(self.A1.T, dZ2) / batch_size
        db2 = np.sum(dZ2, axis=0, keepdims=True) / batch_size
        
        # Hidden layer gradient
        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * (self.Z1 > 0).astype(float)
        dW1 = np.dot(X.T, dZ1) / batch_size
        db1 = np.sum(dZ1, axis=0, keepdims=True) / batch_size
        
        # Update weights
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def train_step(self, X_batch, y_batch, learning_rate=0.1):
        predictions = self.forward(X_batch)
        loss = self.cross_entropy_loss(predictions, y_batch)
        self.backward(X_batch, y_batch, learning_rate)
        return loss, predictions
    
    def predict_class(self, X):
        predictions = self.forward(X)
        return np.argmax(predictions, axis=1)


# ============================================
# LOAD MNIST DATASET
# ============================================
print("="*50)
print("LOADING MNIST DATASET")
print("="*50)

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(f"Original training data shape: {x_train.shape}")  # (60000, 28, 28)
print(f"Original test data shape: {x_test.shape}")      # (10000, 28, 28)

# Preprocess: flatten 28x28 images to 784 pixels
x_train = x_train.reshape(60000, 784) / 255.0  # Normalize to 0-1
x_test = x_test.reshape(10000, 784) / 255.0

print(f"Flattened training data: {x_train.shape}")  # (60000, 784)
print(f"Flattened test data: {x_test.shape}")      # (10000, 784)

# One-hot encode labels (0 → [1,0,0,0,0,0,0,0,0,0])
def one_hot_encode(y, num_classes=10):
    one_hot = np.zeros((len(y), num_classes))
    one_hot[np.arange(len(y)), y] = 1
    return one_hot

y_train_oh = one_hot_encode(y_train)
y_test_oh = one_hot_encode(y_test)

print(f"Training labels shape: {y_train_oh.shape}")  # (60000, 10)
print(f"Test labels shape: {y_test_oh.shape}")      # (10000, 10)

# Show sample of what one-hot encoding looks like
print(f"\nExample: Digit 5 becomes {y_train_oh[5]}")

input_size = 784   # 28×28 pixels
hidden_size = 128  # Hidden layer neurons
output_size = 10   # Digits 0-9

print("\n" + "="*50)
print("CREATING NEURAL NETWORK")
print("="*50)
print(f"Input size: {input_size}")
print(f"Hidden size: {hidden_size}")
print(f"Output size: {output_size}")

nn = NeuralNetwork(input_size, hidden_size, output_size)

# ============================================
# TRAINING LOOP
# ============================================
epochs = 20
batch_size = 64
learning_rate = 0.1

print("\n" + "="*50)
print("TRAINING NEURAL NETWORK ON MNIST")
print("="*50)
print(f"Epochs: {epochs}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Total training samples: {len(x_train)}")
print(f"Batches per epoch: {len(x_train) // batch_size}")
print("\nStarting training...\n")

for epoch in range(epochs):
    # Shuffle training data
    indices = np.random.permutation(len(x_train))
    x_shuffled = x_train[indices]
    y_shuffled = y_train_oh[indices]
    
    total_loss = 0
    num_batches = 0
    
    # Mini-batch training
    for i in range(0, len(x_train), batch_size):
        X_batch = x_shuffled[i:i+batch_size]
        y_batch = y_shuffled[i:i+batch_size]
        
        loss, _ = nn.train_step(X_batch, y_batch, learning_rate)
        total_loss += loss
        num_batches += 1
    
    avg_loss = total_loss / num_batches
    
    # Evaluate on test set every epoch
    predictions = nn.predict_class(x_test)
    accuracy = np.mean(predictions == y_test)
    
    print(f"Epoch {epoch+1:2d}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# ============================================
# FINAL EVALUATION
# ============================================
print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)

final_predictions = nn.predict_class(x_test)
final_accuracy = np.mean(final_predictions == y_test)
print(f"Final test accuracy: {final_accuracy:.4f} ({final_accuracy*100:.2f}%)")

# Show sample predictions
print("\nSample predictions (first 20 test images):")
print("-" * 50)
print("Index | Predicted | Actual | Status")
print("-" * 50)
for i in range(20):
    status = "✓" if final_predictions[i] == y_test[i] else "✗"
    print(f"{i:5d} | {final_predictions[i]:9d} | {y_test[i]:6d} | {status}")

# Show confusion matrix (simplified)
print("\n" + "="*50)
print("CONFUSION MATRIX (Simplified)")
print("="*50)

# Count correct predictions per digit
correct_by_digit = {}
total_by_digit = {}

for i in range(len(x_test)):
    true_digit = y_test[i]
    pred_digit = final_predictions[i]
    
    total_by_digit[true_digit] = total_by_digit.get(true_digit, 0) + 1
    if true_digit == pred_digit:
        correct_by_digit[true_digit] = correct_by_digit.get(true_digit, 0) + 1

print("\nAccuracy per digit:")
print("-" * 30)
for digit in range(10):
    total = total_by_digit.get(digit, 0)
    correct = correct_by_digit.get(digit, 0)
    acc = correct / total if total > 0 else 0
    print(f"Digit {digit}: {correct}/{total} ({acc*100:.1f}%)")

# Show prediction confidence for one sample
print("\n" + "="*50)
print("PREDICTION CONFIDENCE EXAMPLE")
print("="*50)

sample_idx = 0
sample = x_test[sample_idx:sample_idx+1]
probabilities = nn.forward(sample)
print(f"\nFor test image {sample_idx} (Actual digit: {y_test[sample_idx]}):")
print(f"Predicted digit: {final_predictions[sample_idx]}")
print("\nConfidence scores for each digit:")
for digit in range(10):
    bar = "█" * int(probabilities[0][digit] * 50)
    print(f"  {digit}: {probabilities[0][digit]:.4f} {bar}")

# ============================================
# SAVE THE TRAINED MODEL
# ============================================
print("\n" + "="*50)
print("SAVING MODEL")
print("="*50)

np.savez('mnist_weights.npz', 
         W1=nn.W1, b1=nn.b1, 
         W2=nn.W2, b2=nn.b2)

print("✅ Model saved as 'mnist_weights.npz'")

def load_model(filename='mnist_weights.npz'):
    """Load previously saved model weights"""
    data = np.load(filename)
    nn = NeuralNetwork(784, 128, 10)
    nn.W1 = data['W1']
    nn.b1 = data['b1']
    nn.W2 = data['W2']
    nn.b2 = data['b2']
    return nn

print("\nTo load this model later, use:")
print("  nn = load_model('mnist_weights.npz')")

print("\n" + "="*50)
print("✅ MNIST TRAINING COMPLETE!")
print("="*50)