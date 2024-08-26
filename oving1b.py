import pandas as pd
import torch
import matplotlib.pyplot as plt

# Read the csv file of day, length, and weight
data = pd.read_csv('https://gitlab.com/ntnu-tdat3025/regression/childgrowth-datasets/raw/master/day_length_weight.csv', comment='#', header=None, names=['day', 'length', 'weight'])

# Extract the first and second column
days = data['day'].values
lengths = data['length'].values
weights = data['weight'].values

# Convert to torch tensor array format
x_train = torch.tensor(lengths, dtype=torch.float32).reshape(-1, 1)
y_train = torch.tensor(weights, dtype=torch.float32).reshape(-1, 1)
z_train = torch.tensor(days, dtype=torch.float32).reshape(-1, 1)

class LinearRegressionModel3D:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)  # requires_grad enables calculation of gradients
        self.b = torch.tensor([[0.0]], requires_grad=True)

    # Predictor
    def f(self, x, y):
        return torch.cat((x, y), dim=1) @ self.W + self.b  # @ corresponds to matrix multiplication

    # Uses Mean Squared Error
    def loss(self, x, y, z):
        return torch.nn.functional.mse_loss(self.f(x, y), z)

# Declaring an instance of the class LinearRegressionModel
model = LinearRegressionModel3D()

# Declaring learning rate and epochs for the optimization
learning_rate = 0.0001
epochs = 100000

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], learning_rate)
for epoch in range(epochs):
    model.loss(x_train, y_train, z_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train, z_train)))

# Visualize result
fig = plt.figure()
ax = fig.add_subplot(projection='3d')  # 3D plots require a subplot with 3D projection enabled
plt.plot(x_train, y_train, z_train, 'o', label='$(x^{(i)},y^{(i)},z^{(i)})$')
ax.set_xlabel('$x (length)$')
ax.set_ylabel('$y$ (weight)')
ax.set_zlabel('$z$ (days)')

x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
y = torch.tensor([[torch.min(y_train)], [torch.max(y_train)]])

plt.plot(x, y, model.f(x, y).detach(), label='$f(x) = xW+b$', color='black')
plt.legend()
plt.show()
