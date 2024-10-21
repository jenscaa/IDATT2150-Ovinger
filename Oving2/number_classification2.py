import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import matplotlib.pyplot as plt

# Load MNIST data
mnist_train = torchvision.datasets.MNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 784).float()
y_train = mnist_train.targets

mnist_test = torchvision.datasets.MNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 784).float()
y_test = mnist_test.targets

# Define the model using W and b instead of nn.Linear
class SoftmaxModel(nn.Module):
    def __init__(self):
        super(SoftmaxModel, self).__init__()
        # Initialize W (weights) and b (bias) manually
        self.W = nn.Parameter(torch.ones(784, 10))          # Ones initialization of weights
        self.b = nn.Parameter(torch.zeros(10))              # Initialize biases to zero

    def f(self, x):
        # Compute softmax(Wx + b)
        return torch.softmax(x @ self.W + self.b, dim=1)

    def logits(self, x):
        # Return logits (Wx + b) without applying softmax
        return x @ self.W + self.b

    def accuracy(self, x, y):
        # Compute accuracy
        predictions = self.f(x).argmax(dim=1)
        return torch.mean((predictions == y).float())

    def loss(self, x, y):
        # Cross-entropy loss
        return torch.nn.functional.cross_entropy(self.logits(x), y)

# Initialize the model
model = SoftmaxModel()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    optimizer.zero_grad()

    # Compute loss
    loss = model.loss(x_train, y_train)

    # Backpropagation
    loss.backward()
    optimizer.step()

    # Calculate and print accuracy on the test set
    with torch.no_grad():
        test_accuracy = model.accuracy(x_test, y_test)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}, Accuracy: {test_accuracy.item() * 100}%")

    # Early stopping if accuracy is 90% or higher
    if test_accuracy.item() >= 0.91:
        break

# Print final weights and biases
print(f"Final weights W: {model.W.data}")
print(f"Final biases b: {model.b.data}")
print(f"Final Loss: {model.loss(x_train, y_train).item()}, Final Accuracy: {model.accuracy(x_test, y_test).item() * 100}%")

# Visualize and save the weights W after training
weights = model.W.data

for i in range(10):
    weight_image = weights[:, i].reshape(28, 28)
    plt.imshow(weight_image, cmap='viridis')
    plt.title(f'Weights for digit {i}')
    plt.colorbar()
    plt.savefig(f'weight_digit_{i}.png')
    plt.close()
