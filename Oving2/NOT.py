import matplotlib.pyplot as plt
import numpy as np
import torch

W_init = np.array([[0.0]])
b_init = np.array([[0.0]])

# Da Sigmoid function!
def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

# The NOT class
class NotOperatorModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = torch.tensor(W, requires_grad=True, dtype=torch.float32)
        self.b = torch.tensor(b, requires_grad=True, dtype=torch.float32)

    def f(self, x):
        return sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        f_x = self.f(x)
        return -torch.mean(y * torch.log(f_x) + (1 - y) * torch.log(1 - f_x))

model = NotOperatorModel()

# Training input and output
x_train = torch.tensor([[0], [1]], dtype=torch.float32)
y_train = torch.tensor([[1], [0]], dtype=torch.float32)

# Declaring learning rate and epochs for the optimization
learning_rate = 0.1
epochs = 150

# Using Adam optimizer
optimizer = torch.optim.Adam([model.W, model.b], lr=learning_rate)

for epoch in range(epochs):
    loss_value = model.loss(x_train, y_train)
    loss_value.backward()   # Compute loss gradients
    optimizer.step()        # Perform optimization by adjusting W and b,
    optimizer.zero_grad()   # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))
print("f(x) = %sx + %s\n" % (model.W.item(), model.b.item()))

# Test this bad boy to see if it works
test_data = torch.tensor([[0], [1], [0], [0.9], [0.7], [0.51], [0.2], [1], [0.1], [0.6], [0.49]], dtype=torch.float32)
for x_i in test_data.data:
    not_value = round(model.f(x_i).item(), 2)
    print(f"NOT({round(x_i.item(), 2)}) = {not_value} \u2248 {round(not_value, 0)}")

# Visualize result
plt.title('NOT-operator', fontweight='bold', fontsize=17)
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label=f'$f(x) = {round(model.W.item(), 2)}x+{round(model.b.item(), 2)}$', color='green')
plt.legend()
plt.show()