import torch
import gymnasium as gym
import numpy as np
from Oving8.QNetwork import QNetwork

# Define a function to select the best action (exploitation)
def select_action(state, qnetwork):
    state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        q_values = qnetwork(state)
    return np.argmax(q_values.numpy())  # Return the action with the highest Q-value

# Load the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")

# Get state and action sizes from the environment
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Initialize the Q-network
qnetwork = QNetwork(state_size, action_size)

# Load the saved Q-network weights
qnetwork.load_state_dict(torch.load('trained_cart_pole.pth'))
qnetwork.eval()  # Set the network to evaluation mode (no training)

# Run the environment using the trained agent
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    action = select_action(state, qnetwork)  # Select the best action
    next_state, reward, done, truncated, _ = env.step(action)
    state = next_state
    total_reward += reward
    env.render()  # Render the environment

print(f"Total Reward: {total_reward}")
env.close()
