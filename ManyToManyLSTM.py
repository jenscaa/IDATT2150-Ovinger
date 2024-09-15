import torch
import torch.nn as nn

# Modifisert LSTM-modell
class LongShortTermMemoryModel(nn.Module):

    def __init__(self, encoding_size):
        super(LongShortTermMemoryModel, self).__init__()

        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, encoding_size)  # 128 is the state size

    def reset(self):  # Reset states prior to new input sequence
        zero_state = torch.zeros(1, 1, 128)  # Shape: (number of layers, batch size, state size)
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):  # x shape: (sequence length, batch size, encoding size)
        out, (self.hidden_state, self.cell_state) = self.lstm(x, (self.hidden_state, self.cell_state))
        return self.dense(out.reshape(-1, 128))

    def f(self, x):  # x shape: (sequence length, batch size, encoding size)
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):  # x shape: (sequence length, batch size, encoding size), y shape: (sequence length, encoding size)
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))

# Utvidet tegnkoding for "hello world"
char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'w'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'd'
]

index_to_char = [' ', 'h', 'e', 'l', 'o', 'w', 'r', 'd']

# Treningsdata for "hello world"
x_train = torch.tensor([
    [char_encodings[1]],  # 'h'
    [char_encodings[2]],  # 'e'
    [char_encodings[3]],  # 'l'
    [char_encodings[3]],  # 'l'
    [char_encodings[4]],  # 'o'
    [char_encodings[0]],  # ' '
    [char_encodings[5]],  # 'w'
    [char_encodings[4]],  # 'o'
    [char_encodings[6]],  # 'r'
    [char_encodings[3]],  # 'l'
    [char_encodings[7]]   # 'd'
])

y_train = torch.tensor([
    char_encodings[2],  # 'e'
    char_encodings[3],  # 'l'
    char_encodings[3],  # 'l'
    char_encodings[4],  # 'o'
    char_encodings[0],  # ' '
    char_encodings[5],  # 'w'
    char_encodings[4],  # 'o'
    char_encodings[6],  # 'r'
    char_encodings[3],  # 'l'
    char_encodings[7],  # 'd'
    char_encodings[0],  # ' '
])

encoding_size = len(char_encodings[0])
model = LongShortTermMemoryModel(encoding_size)

optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

# Trening av modellen
for epoch in range(1000):
    model.reset()
    loss = model.loss(x_train, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if epoch % 100 == 99:
        # Generer tegnsekvens fra input 'h'
        model.reset()
        text = 'h'
        y = model.f(torch.tensor([[char_encodings[1]]]))  # Starter med 'h'
        text += index_to_char[y.argmax(1)]

        for c in range(50):  # Generer 50 tegn
            y = model.f(torch.tensor([[char_encodings[y.argmax(1)]]]))
            text += index_to_char[y.argmax(1)]
        print(f'Epoch {epoch + 1}: {text}')
