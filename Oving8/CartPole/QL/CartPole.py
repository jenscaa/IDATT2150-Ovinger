import numpy as np
import gymnasium as gym
import pickle  # For loading the Q-table

# Load the saved Q-table
with open('q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

# Create the CartPole environment
env = gym.make("CartPole-v1", render_mode="human")  # Use "human" to render the environment

# Discretization bins (must match the original)
num_bins = 20

def create_bins():
    bins = [
        np.linspace(-4.8, 4.8, num_bins),  # Cart position
        np.linspace(-4, 4, num_bins),  # Cart velocity
        np.linspace(-0.418, 0.418, num_bins),  # Pole angle
        np.linspace(-4, 4, num_bins)  # Pole angular velocity
    ]
    return bins

# Discretize the continuous state space into bins
def discretize_state(state, bins):
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], bins[i]) - 1)  # Convert to discrete
    return tuple(state_disc)

# Initialize the bins (same as the training script)
bins = create_bins()

# Test the trained agent by exploiting the Q-table
state, _ = env.reset()
state = discretize_state(state, bins)
done = False
total_reward = 0

while not done:
    # Choose the best action (exploit the trained policy)
    action = np.argmax(q_table[state])
    next_state, reward, done, truncated, _ = env.step(action)
    next_state = discretize_state(next_state, bins)
    state = next_state
    total_reward += reward

    # Render the environment (can be slow, but allows you to visualize)
    env.render()

print(f"Total Reward: {total_reward}")
env.close()
