import torch
import torch.nn as nn

index_to_char = [' ', 'a', 'c', 'e', 'f', 'h', 'l', 'm', 'n', 'o', 'p', 'r', 's', 't']
char_encodings = [
    [1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # ' '
    [0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'a'
    [0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'c'
    [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'e'
    [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'f'
    [0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],  # 'h'
    [0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],  # 'l'
    [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],  # 'm'
    [0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],  # 'n'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],  # 'o'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],  # 'p'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],  # 'r'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],  # 's'
    [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],  # 't'
]

word_to_emoji = {
    "hat ": '🎩',
    "rat ": '🐀',
    "cat ": '🐱',
    "flat": '🏠',
    "matt": '🧔',
    "cap ": '🧢',
    "son ": '👦'
}

emoji_list = ['🎩', '🐀', '🐱', '🏠', '🧔', '🧢', '👦']
emoji_to_idx = {emoji: i for i, emoji in enumerate(emoji_list)}


def one_hot_encode_char(char):
    index = index_to_char.index(char)
    return torch.tensor(char_encodings[index])

# Preparing training data
def prepare_data(word_to_emoji, max_word_length):
    X_train, y_train = [], []
    for word, emoji in word_to_emoji.items():
        padded_word = word.rjust(max_word_length)  # Pad words with spaces
        encoded_word = [one_hot_encode_char(c) for c in padded_word]  # One-hot encode characters
        X_train.append(torch.stack(encoded_word))
        y_train.append(emoji_to_idx[emoji])  # Emoji as label
    return torch.stack(X_train), torch.tensor(y_train)

# Define the many-to-one LSTM model
class ManyToOneLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ManyToOneLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        _, (output, _) = self.lstm(x)     # Only take the last output
        out = self.fc(output.squeeze(0))  # Many-to-one: one output per sequence
        return out

# Model parameters
max_word_length = 4  # Max word length (including padding)
input_size = len(char_encodings[0])  # One-hot encoded vector size (char set size)
hidden_size = 64  # Hidden state size
output_size = len(emoji_list)  # Number of emojis
# Create the model
model = ManyToOneLSTM(input_size, hidden_size, output_size)

# Optimize model
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()
# Prepare training data
X_train, y_train = prepare_data(word_to_emoji, max_word_length)
epochs = 1000
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Testing function to predict emoji for a given word
def predict_emoji(word):
    padded_word = word.rjust(max_word_length)  # Pad word if necessary
    encoded_word = [one_hot_encode_char(c) for c in padded_word]
    input_seq = torch.stack(encoded_word).unsqueeze(0)  # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(input_seq)
        predicted_idx = torch.argmax(output, dim=1).item()
        return emoji_list[predicted_idx]

# Test the model yourself
word = input("Test yourself: ")
while word != "q":
    try:
        predicted_emoji = predict_emoji(word)
        print(f"Word: {word}, Predicted Emoji: {predicted_emoji}")
    except ValueError as e:
        print(f"Error: The word '{word}' contains invalid characters because {e}.")
    word = input("Test yourself: ")

