import numpy as np
import gymnasium as gym
import random
import pickle

# Create the CartPole environment
env = gym.make("CartPole-v1")

# Hyperparameters
alpha = 0.1  # Lower learning rate for more stable learning
gamma = 0.95  # Discount factor
epsilon = 1.0  # Initial epsilon for exploration
epsilon_decay = 0.999  # Slower decay rate for epsilon
min_epsilon = 0.01 # Higher minimum epsilon to encourage exploration
num_bins = 20  # More bins to capture more granular state differences
num_episodes = 10000  # Number of episodes to train
max_steps = 500  # Increased max steps per episode

# Discretization bins
def create_bins():
    bins = [
        np.linspace(-4.8, 4.8, num_bins),  # Cart position
        np.linspace(-4, 4, num_bins),  # Cart velocity
        np.linspace(-0.418, 0.418, num_bins),  # Pole angle
        np.linspace(-1_000_000, 1_000_000, num_bins)  # Pole angular velocity
    ]
    return bins

# Discretize the continuous state space into bins
def discretize_state(state, bins):
    state_disc = []
    for i in range(len(state)):
        state_disc.append(np.digitize(state[i], bins[i]) - 1)  # Convert to discrete
    return tuple(state_disc)

# Q-table initialization
bins = create_bins()
q_table = np.zeros([num_bins] * 4 + [env.action_space.n])  # 4 dimensions for state variables, 1 for actions

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Exploration
    else:
        return np.argmax(q_table[state])  # Exploitation

# Initialize variables to track the best performance
best_total_reward = -float('inf')  # Set to negative infinity to ensure the first reward is better
best_q_table = None  # Will hold the best Q-table

# Q-learning algorithm
for episode in range(num_episodes):
    state, _ = env.reset()
    state = discretize_state(state, bins)
    total_reward = 0

    for step in range(max_steps):
        action = choose_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = discretize_state(next_state, bins)

        # Q-value update
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state][best_next_action]
        td_error = td_target - q_table[state][action]
        q_table[state][action] += alpha * td_error

        state = next_state
        total_reward += reward

        if done or truncated:
            break

    # Decay epsilon
    if epsilon > min_epsilon:
        epsilon *= epsilon_decay

    # Save the Q-table if this episode's total reward is the best so far
    if total_reward > best_total_reward:
        best_total_reward = total_reward
        best_q_table = q_table.copy()  # Make a copy of the current Q-table
        print(f"New best total reward: {best_total_reward}, saving Q-table...")

        # Save the best Q-table to a file
        with open('q_table.pkl', 'wb') as f:
            pickle.dump(best_q_table, f)

    if episode % 1000 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon}")

print(f"Best total reward during training: {best_total_reward}")
