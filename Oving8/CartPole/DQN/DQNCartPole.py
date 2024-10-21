import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
from Oving8.DQNAgent import DQNAgent

class DQNCartPoolAgent(DQNAgent):
    def __init__(self, state_size, action_size, buffer_size=10000, batch_size=64, gamma=0.99, lr=1e-3, tau=1e-3,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        # Call the constructor of the parent class (DQNAgent)
        super().__init__(state_size, action_size, buffer_size, batch_size, gamma, lr, tau, epsilon, epsilon_min, epsilon_decay)

    def learn(self, experiences):
        """Update Q-Network from a batch of experiences."""
        # Unpack the sampled experiences into states, actions, rewards, next states, and dones
        states, actions, rewards, next_states, dones = experiences

        # Extract cart positions from the states (assuming cart position is the first feature)
        cart_positions = states[:, 0]  # Assuming cart position is at index 0 of the state array

        # Apply the penalty based on the cart position
        penalties = abs(cart_positions * 1.0)  # Penalize based on distance from center (Cart Position = 0)
        rewards = rewards - penalties  # Subtract the penalty from the rewards

        # Convert the experiences to PyTorch tensors
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)  # Add an extra dimension to rewards for batch processing
        dones = torch.FloatTensor(dones).unsqueeze(1)  # Same for dones

        actions = torch.LongTensor(actions).unsqueeze(1)  # Actions need to be long tensors and have an extra dimension

        # Get the Q-values for the current states and actions
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


# Train the DQN Agent on CartPole
env = gym.make("CartPole-v1")

state_size = env.observation_space.shape[0]
action_size = env.action_space.n

agent = DQNCartPoolAgent(state_size, action_size)

num_episodes = 1000
max_steps = 500
scores = []

for episode in range(num_episodes):
    state, _ = env.reset()
    total_reward = 0
    for step in range(max_steps):
        action = agent.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
        if done or truncated:
            break
    scores.append(total_reward)

    if total_reward == 500:
        break

    if episode % 100 == 0:
        print(f"Episode {episode}, Average Reward: {np.mean(scores[-100:])}")


torch.save(agent.qnetwork_local.state_dict(), 'trained_cart_pole.pth')
print("Trained Q-network saved to trained_cart_pole.pth")
env.close()