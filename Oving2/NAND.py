import matplotlib.pyplot as plt
import numpy as np
import torch

W_init = np.array([[0.0], [0.0]])
b_init = np.array([[0.0]])

# Da Sigmoid function!
def sigmoid(t):
    return 1 / (1 + torch.exp(-t))

# The NAND class
class NANDOperatorModel:
    def __init__(self, W=W_init.copy(), b=b_init.copy()):
        self.W = torch.tensor(W, requires_grad=True, dtype=torch.float32)
        self.b = torch.tensor(b, requires_grad=True, dtype=torch.float32)

    def f(self, x):
        x = torch.tensor(x, requires_grad=True, dtype=torch.float32)
        return sigmoid(x @ self.W + self.b)

    def loss(self, x, y):
        f_x = self.f(x)
        return -torch.mean(y * torch.log(f_x) + (1 - y) * torch.log(1 - f_x))

model = NANDOperatorModel()

# Training input and output
x_train = torch.tensor([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[1], [1], [1], [0]], dtype=torch.float32)

# Declaring learning rate and epochs for the optimization
learning_rate = 0.1
epochs = 1000

# Using Adam optimizer
optimizer = torch.optim.Adam([model.W, model.b], lr=learning_rate)

for epoch in range(epochs):
    loss_value = model.loss(x_train, y_train)
    loss_value.backward()   # Compute loss gradients
    optimizer.step()        # Perform optimization by adjusting W and b,
    optimizer.zero_grad()   # Clear gradients for next step

# Print model variables and loss
print("W = %s, b = %s, loss = %s" % (model.W, model.b, model.loss(x_train, y_train)))

# Test this bad boy to see if it works
test_data = torch.tensor([[1, 1], [0, 0], [1, 0], [0, 1], [0.9, 0.1], [0.7, 0.3], [0.51, 0.49], [0.51, 0.51]], dtype=torch.float32)
for x_i in test_data:
    x_vectors = x_i.tolist()
    nand_value = round(model.f(x_i).item(), 2)
    print(f"NAND({round(x_vectors[0], 2)}, {round(x_vectors[1], 2)}) = {nand_value} \u2248 {round(nand_value, 0)}")

# Visualize result
fig = plt.figure("Logistic regression: the logical NAND operator")
plot1 = fig.add_subplot(111, projection='3d')

plot1_f = plot1.plot_wireframe(np.array([[]]), np.array([[]]), np.array([[]]), color="green", label="$f(\\mathbf{x})=\\sigma(\\mathbf{xW}+b)$")

plot1.plot(x_train[:, 0].squeeze(), x_train[:, 1].squeeze(), y_train[:, 0].squeeze(), 'o', label="$(x_1^{(i)}, x_2^{(i)},y^{(i)})$", color="blue")

plot1_info = fig.text(0.01, 0.02, "")

plot1.set_xlabel("$x_1$")
plot1.set_ylabel("$x_2$")
plot1.set_zlabel("$y$")
plot1.legend(loc="upper left")
plot1.set_xticks([0, 1])
plot1.set_yticks([0, 1])
plot1.set_zticks([0, 1])
plot1.set_xlim(-0.25, 1.25)
plot1.set_ylim(-0.25, 1.25)
plot1.set_zlim(-0.25, 1.25)

table = plt.table(
    cellText=[[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0]],
    colWidths=[0.1] * 3,
    colLabels=["$x_1$", "$x_2$", "$f(\\mathbf{x})$"],
    cellLoc="center",
    bbox=[1.0, 0.0, 0.3, 0.3]
)


def update_figure(event=None):
    if (event is not None):
        if event.key == "W":
            model.W[0, 0] += 0.2
        elif event.key == "w":
            model.W[0, 0] -= 0.2
        elif event.key == "E":
            model.W[1, 0] += 0.2
        elif event.key == "e":
            model.W[1, 0] -= 0.2

        elif event.key == "B":
            model.b[0, 0] += 0.2
        elif event.key == "b":
            model.b[0, 0] -= 0.2

        elif event.key == "c":
            model.W = W_init.copy()
            model.b = b_init.copy()

    global plot1_f
    plot1_f.remove()
    x1_grid, x2_grid = np.meshgrid(np.linspace(-0.25, 1.25, 10), np.linspace(-0.25, 1.25, 10))
    y_grid = np.empty([10, 10])
    for i in range(0, x1_grid.shape[0]):
        for j in range(0, x1_grid.shape[1]):
            y_grid[i, j] = model.f([[x1_grid[i, j], x2_grid[i, j]]])
    plot1_f = plot1.plot_wireframe(x1_grid, x2_grid, y_grid, color="green")

    plot1_info.set_text(
        "$\\mathbf{W}=\\genfrac{[}{]}{0}{}{%.2f}{%.2f}$\n$b=[%.2f]$\n$loss = -\\frac{1}{N}\\sum_{i=1}^{N}\\left [ y^{(i)} \\log\\/f(\\mathbf{x}^{(i)}) + (1-y^{(i)}) \\log (1-f(\\mathbf{x}^{(i)})) \\right ] = %.2f$"
        % (model.W[0, 0], model.W[1, 0], model.b[0, 0], model.loss(x_train, y_train))
    )

    table._cells[(1, 2)]._text.set_text("${%.1f}$" % model.f([[0, 0]]))
    table._cells[(2, 2)]._text.set_text("${%.1f}$" % model.f([[0, 1]]))
    table._cells[(3, 2)]._text.set_text("${%.1f}$" % model.f([[1, 0]]))
    table._cells[(4, 2)]._text.set_text("${%.1f}$" % model.f([[1, 1]]))

    plt.pause(0.01)
    fig.canvas.draw()


update_figure()
fig.canvas.mpl_connect('key_press_event', update_figure)

plt.show()