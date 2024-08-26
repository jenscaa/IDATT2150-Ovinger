import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Read the csv file of day and head circumference
data = pd.read_csv('https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_head_circumference.csv', comment='#', header=None, names=['day', 'head_circumference'])

# Extract the first and second column
days = data['day'].values
head_circumferences = data['head_circumference'].values

# Convert to torch tensor array format
x_train = torch.tensor(days, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(head_circumferences, dtype=torch.float32).reshape(-1, 1)

class NotLinearRegressionModel2D:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x):
        return 20 * 1 / (1 + torch.exp(-(x * self.W + self.b))) + 31

    # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.nn.functional.mse_loss(self.f(x), y)

# Declaring an instance of the class NotLinearRegressionModel2D
model = NotLinearRegressionModel2D()

# Declaring learning rate and epochs for the optimization
learning_rate = 1e-13
epochs = 1100

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
for epoch in range(epochs):
    loss = model.loss(x_train, y_train)
    loss.backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W.item(), model.b.item(), model.loss(x_train, y_train).item()))

# Visualize result
plt.plot(x_train.numpy(), y_train.numpy(), 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('Day')
plt.ylabel('Head Circumference')

# Generate data for plotting the model
x = np.linspace(0, x_train.max().item() + 5, 400)  # NumPy array for plotting
y = 20*1/(1+np.exp(-(x*model.W.item() + model.b.item()))) + 31

plt.plot(x, y, label='$f(x)$', color='black')  # Plot the function
plt.legend()
plt.show()
