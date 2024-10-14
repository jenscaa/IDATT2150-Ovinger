import numpy as np
import gymnasium as gym

# Initialise the environment
env = gym.make("CartPole-v1", render_mode="human")


# Hyperparameters for Q-learning
alpha = 0.1   # Learning rate
gamma = 0.95  # Discount factor
epsilon = 1.0  # Exploration rate
epsilon_min = 0.01
epsilon_decay = 0.995
num_episodes = 500

# Number of bins to discretize each continuous state variable
num_bins = [6, 6, 12, 12]  # 6 bins for cart position/velocity, 12 for angle/angular velocity

# Check if observation_space is of type Box
if isinstance(env.observation_space, gym.spaces.Box):
    # Define the upper and lower bounds for each state variable
    state_bounds = list(zip(env.observation_space.low, env.observation_space.high))

    # Modify the velocity and angular velocity limits manually for better discretization
    state_bounds[1] = [-6, 6]  # Cart velocity limits
    state_bounds[3] = [-7, 7]  # Pole angular velocity limits
else:
    raise ValueError("Expected observation_space to be of type Box.")

# Print the state bounds to verify
print("State bounds:", state_bounds)


# Discretize the continuous state space
def discretize_state(state, num_bins, state_bounds):
    discrete_state = []
    for i in range(len(state)):
        low, high = state_bounds[i]
        if state[i] <= low:
            bin_idx = 0
        elif state[i] >= high:
            bin_idx = num_bins[i] - 1
        else:
            # Scale continuous state to discrete bins
            bin_idx = int((state[i] - low) / (high - low) * num_bins[i])
        discrete_state.append(bin_idx)
    return tuple(discrete_state)

# Initialize Q-table (state bins + actions)
q_table = np.zeros(num_bins + [env.action_space.n])

# Epsilon-greedy action selection
def choose_action(state, epsilon):
    if np.random.random() < epsilon:
        return env.action_space.sample()  # Explore
    else:
        return np.argmax(q_table[state])  # Exploit

# Reset the environment to generate the first observation
state, info = env.reset(seed=42)

# Q-learning algorithm
for episode in range(num_episodes):
    # Reset environment for each episode
    state, info = env.reset(seed=42)
    state = discretize_state(state, num_bins, state_bounds)

    done = False
    total_reward = 0

    while not done:
        # Choose action using epsilon-greedy policy
        action = choose_action(state, epsilon)

        # Step through the environment
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Discretize the next state
        next_state = discretize_state(next_state, num_bins, state_bounds)

        # Q-value update
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (
                    reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])

        # Move to the next state
        state = next_state
        total_reward += reward

        if done:
            break

    # Decay epsilon to reduce exploration over time
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    print(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

env.close()