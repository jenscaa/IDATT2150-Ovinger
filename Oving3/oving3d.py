import torch
import torch.nn as nn
import torchvision

# Load observations from the mnist dataset. The observations are divided into a training set and a test set
mnist_train = torchvision.datasets.FashionMNIST('./data', train=True, download=True)
x_train = mnist_train.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_train = torch.zeros((mnist_train.targets.shape[0], 10))  # Create output tensor
y_train[torch.arange(mnist_train.targets.shape[0]), mnist_train.targets] = 1  # Populate output

mnist_test = torchvision.datasets.FashionMNIST('./data', train=False, download=True)
x_test = mnist_test.data.reshape(-1, 1, 28, 28).float()  # torch.functional.nn.conv2d argument must include channels (1)
y_test = torch.zeros((mnist_test.targets.shape[0], 10))  # Create output tensor
y_test[torch.arange(mnist_test.targets.shape[0]), mnist_test.targets] = 1  # Populate output

# Normalization of inputs
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# Divide training data into batches to speed up optimization
batches = 600
x_train_batches = torch.split(x_train, batches)
y_train_batches = torch.split(y_train, batches)


class ConvolutionalNeuralNetworkModel(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetworkModel, self).__init__()

        # First convolutional layer (1 input channel, 32 output channels, 5x5 kernel, padding=2)
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)

        # First max-pooling layer (reduces 28x28 to 14x14)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        # Second convolutional layer (32 input channels, 64 output channels, 5x5 kernel, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)

        # Second max-pooling layer (reduces 14x14 to 7x7)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Fully connected layer (flattened input of size 64 * 7 * 7 mapped to 1024 units)
        self.dense1 = nn.Linear(64 * 7 * 7, 1024)

        # Fully connected layer (1024 units mapped to 10 output classes)
        self.dense2 = nn.Linear(1024, 10)

        # ReLU activation
        self.relu = nn.ReLU()

    def logits(self, x):
        # Apply first convolution, then ReLU, then pooling
        x = self.pool1(self.relu(self.conv1(x)))

        # Apply second convolution, then ReLU, then pooling
        x = self.pool2(self.relu(self.conv2(x)))

        # Flatten the feature maps for the fully connected layer
        x = x.view(-1, 64 * 7 * 7)  # Reshape the tensor to (batch_size, 3136)

        # Apply the first fully connected layer (to 1024 units) with ReLU
        x = self.relu(self.dense1(x))

        # Apply the second fully connected layer (to 10 output units) without ReLU (final layer)
        return self.dense2(x)

    # Predictor function (applies softmax to logits)
    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    # Loss function (cross-entropy)
    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

    # Accuracy function
    def accuracy(self, x, y):
        return torch.mean(torch.eq(self.f(x).argmax(1), y.argmax(1)).float())


model = ConvolutionalNeuralNetworkModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.Adam(model.parameters(), 0.001)
for epoch in range(20):
    for batch in range(len(x_train_batches)):
        model.loss(x_train_batches[batch], y_train_batches[batch]).backward()  # Compute loss gradients
        optimizer.step()  # Perform optimization by adjusting W and b,
        optimizer.zero_grad()  # Clear gradients for next step

    print('accuracy = %f%%' % (model.accuracy(x_test, y_test).item() * 100))
