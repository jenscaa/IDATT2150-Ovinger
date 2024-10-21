import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from QNetwork import QNetwork
from Memory import ReplayBuffer

# DQN Agent
class DQNAgent:
    # The constructor initializes important hyperparameters and sets up the neural networks, optimizer, and replay buffer
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, tau=1e-3,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        # Dimensions of the state and action space
        self.state_size = state_size
        self.action_size = action_size

        # Discount factor: how much the agent cares about future rewards
        self.gamma = gamma

        # Tau: Used for soft updating of the target Q-network
        self.tau = tau

        # Epsilon: Initial value for the epsilon-greedy policy (high exploration at start)
        self.epsilon = epsilon

        # Minimum value for epsilon (low exploration at the end of training)
        self.epsilon_min = epsilon_min

        # Epsilon decay: Controls how fast epsilon decays (shifts from exploration to exploitation)
        self.epsilon_decay = epsilon_decay

        # Initialize the Q-networks
        # The local network is used to select actions, while the target network is used to compute stable target Q-values
        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)

        # Optimizer: Used to update the weights of the local Q-network (Adam optimizer)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        # Replay buffer: Stores past experiences (state, action, reward, next state, done)
        self.memory = ReplayBuffer(buffer_size, batch_size)

        # Store the batch size for training
        self.batch_size = batch_size

    # This method selects an action using an epsilon-greedy strategy
    def act(self, state):
        """Returns actions for a given state as per the current policy."""
        # Convert the state to a PyTorch tensor and add a batch dimension with unsqueeze(0)
        state = torch.FloatTensor(state).unsqueeze(0)

        # With probability epsilon, select a random action (exploration)
        if np.random.rand() <= self.epsilon:
            return np.random.randint(self.action_size)
        else:
            # Otherwise, select the best action based on the Q-network (exploitation)
            with torch.no_grad():  # No gradient calculations needed for inference
                q_values = self.qnetwork_local(state)
            return np.argmax(q_values.numpy())  # Select action with the highest Q-value

    # This method stores the experience and triggers learning if enough experiences are available
    def step(self, state, action, reward, next_state, done):
        """Stores experience in replay buffer and learns every few steps."""
        # Add the experience (state, action, reward, next_state, done) to the replay buffer
        self.memory.add((state, action, reward, next_state, done))

        # If there are enough experiences in memory, sample a batch and update the Q-network
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

    # This method updates the Q-network using experiences sampled from the replay buffer
    def learn(self, experiences):
        """Update Q-Network from a batch of experiences."""
        # Unpack the sampled experiences into states, actions, rewards, next states, and dones
        states, actions, rewards, next_states, dones = experiences

        # Convert the experiences to PyTorch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Add an extra dimension to rewards for batch processing
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Same for dones

        actions = torch.LongTensor(actions).unsqueeze(1)  # Actions need to be long tensors and have an extra dimension

        # Get the Q-values for the current states and actions
        # `gather(1, actions)` selects the Q-value for the action taken from the batch of Q-values
        q_values = self.qnetwork_local(states).gather(1, actions)

        # Get the max Q-values for the next states from the target network (no gradient calculation)
        next_q_values = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute the target Q-values: reward + discounted max Q-value from next state (if not done)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute the loss (mean squared error between current Q-values and target Q-values)
        loss = nn.MSELoss()(q_values, target_q_values)

        # Zero the gradients to prevent accumulation and backpropagate the loss to compute gradients
        self.optimizer.zero_grad()
        loss.backward()

        # Update the weights of the local Q-network using the optimizer
        self.optimizer.step()

        # Update the target Q-network using soft updates (slowly updating it to follow the local Q-network)
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

        # Decrease epsilon to reduce the amount of exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # Soft update of the target Q-network: gradually update the target network towards the local network
    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # target_param = tau * local_param + (1 - tau) * target_param
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
