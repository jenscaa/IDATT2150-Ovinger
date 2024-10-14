import gymnasium as gym
import torch
from sympy.codegen.futils import render_as_module

from DQNAgent import DQNAgent



# Main function to train the DQN agent over a number of episodes
def train_dqn_agent(episodes=1000):
    # Initialize the LunarLander environment
    env = gym.make("LunarLander-v3", render_mode="human")

    # Get the dimensions of the state and action spaces
    state_size = env.observation_space.shape[0]  # State size is typically 8 for LunarLander
    action_size = env.action_space.n  # Action size is 4 for LunarLander (4 discrete actions)

    # Initialize the DQN agent with the state and action sizes
    agent = DQNAgent(state_size, action_size, batch_size=128)

    # Run through the specified number of episodes
    for episode in range(episodes):
        # Reset the environment to get the initial state
        state, _ = env.reset()

        # Initialize total reward for the episode
        total_reward = 0
        done = False

        # Run the episode until the agent reaches a terminal state
        while not done:
            # Agent selects an action based on the current state
            action = agent.act(state)

            # Take the action and receive the next state, reward, and done flag
            next_state, reward, done, truncated, info = env.step(action)

            # Store the experience and update the agent
            agent.step(state, action, reward, next_state, done)

            # Move to the next state
            state = next_state

            # Accumulate the reward for this episode
            total_reward += reward

            # If the episode is done, break the loop
            if done:
                print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")
                break

    # Close the environment after training
    env.close()

    # After the training is complete, save the model
    save_trained_agent(agent, filename="trained_lunar_lander.pth")


# After training, save the local Q-network's weights
def save_trained_agent(agent, filename="trained_lunar_lander.pth"):
    torch.save(agent.qnetwork_local.state_dict(), filename)
    print(f"Model saved to {filename}")


# Run the training loop
if __name__ == "__main__":
    train_dqn_agent()