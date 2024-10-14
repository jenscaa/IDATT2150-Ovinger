import gymnasium as gym
import torch
import torch.nn as nn
import numpy as np
from QNetwork import QNetwork

# Load the trained agent and run it in the environment with rendering
def run_trained_agent(filename="trained_lunar_lander.pth"):
    # Load the LunarLander-v3 environment with rendering
    env = gym.make("LunarLander-v3", render_mode="human")

    # Get state and action sizes
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Initialize the Q-network and load the saved model weights
    qnetwork_local = QNetwork(state_size, action_size)
    qnetwork_local.load_state_dict(torch.load(filename))
    qnetwork_local.eval()  # Set the model to evaluation mode (no training)

    # Reset the environment to get the initial state
    state, _ = env.reset()

    # Run the agent in a loop until the episode ends
    done = False
    total_reward = 0

    while not done:
        # Convert the state to a tensor and pass it through the trained Q-network
        state = torch.FloatTensor(state).unsqueeze(0)

        # Select the action with the highest Q-value
        with torch.no_grad():
            q_values = qnetwork_local(state)
            action = np.argmax(q_values.numpy())  # Choose the action with the highest Q-value

        # Take the action and render the environment
        next_state, reward, done, truncated, info = env.step(action)

        # Accumulate reward for tracking purposes
        total_reward += reward

        # Set the new state for the next iteration
        state = next_state

    # Print the total reward received during the episode
    print(f"Total Reward: {total_reward}")

    # Close the environment after the episode finishes
    env.close()

# Example usage:
run_trained_agent("trained_lunar_lander.pth")
