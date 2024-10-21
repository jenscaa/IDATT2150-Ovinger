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

# Classification model with multiple layers
class MultiLayerModel(nn.Module):
    def __init__(self):
        super(MultiLayerModel, self).__init__()
        self.layer1 = nn.Linear(784, 256)  # First hidden layer
        self.layer2 = nn.Linear(256, 128)  # Second hidden layer
        self.layer3 = nn.Linear(128, 10)   # Output layer

    def f(self, x):
        x = torch.relu(self.layer1(x))     # Apply ReLU activation after first layer
        x = torch.relu(self.layer2(x))     # Apply ReLU activation after second layer
        return torch.softmax(self.layer3(x), dim=1)  # Softmax activation at the output layer

    def logits(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return self.layer3(x)  # logits (raw scores) from the final layer

    def accuracy(self, x, y):
        predictions = self.f(x).argmax(dim=1)
        return torch.mean((predictions == y).float())

    def loss(self, x, y):
        return torch.nn.functional.cross_entropy(self.logits(x), y)

model = MultiLayerModel()

# Declaring learning rate and epochs for the optimization
epochs = 1000
learning_rate = 0.001

# Using Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    optimizer.zero_grad()
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()

    # Calculate and print accuracy on the test set
    with torch.no_grad():
        test_accuracy = model.accuracy(x_test, y_test)
        print(f"Epoch {epoch + 1}/{epochs}, {' '*(4 - len(str(epoch)))}"
              f"Loss: {loss.item()}, {' '*(20 - len(str(loss.item())))}"
              f"Accuracy: {test_accuracy.item() * 100}%")

    # Early stopping if accuracy is 96% or higher
    if test_accuracy.item() >= 0.96:
        break

print(f"Final weights and biases: {model.layer3.weight.data}, {model.layer3.bias.data}")
print(f"Final Loss: {model.loss(x_train, y_train).item()}, Final Accuracy: {model.accuracy(x_test, y_test).item() * 100}%")

# Visualize and save the weights W of the output layer after training
weights = model.layer1.weight.data

for i in range(10):
    weight_image = weights[i, :].reshape(28, 28)
    plt.imshow(weight_image, cmap='viridis')
    plt.title(f'Weights for digit {i}')
    plt.colorbar()
    plt.savefig(f'W{i}.png')
    plt.close()
