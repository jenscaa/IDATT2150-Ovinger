import pandas as pd
import torch
import matplotlib.pyplot as plt

# Read the csv file of length and weight
data = pd.read_csv('https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/length_weight.csv', comment='#', header=None, names=['length', 'weight'])

# Extract the first and second column
lengths = data['length'].values
weights = data['weight'].values

# Convert to torch tensor array format
x_train = torch.tensor(lengths, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel2D:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return x @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

# Declaring an instance of the class LinearRegressionModel
model = LinearRegressionModel2D()

# Declaring learning rate and epochs for the optimization
learning_rate = 0.0001
epochs = 100000

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
print("f(x) = x%s + %s" % (model.W.item(), model.b.item()))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Length')
plt.ylabel('Weight')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$', color='black')
plt.legend()
plt.show()
