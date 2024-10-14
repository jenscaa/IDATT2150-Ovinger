import numpy as np
from collections import deque
import random

# Replay memory to store experiences
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    # Add a new experience (state, action, reward, next_state, done) to the buffer
    def add(self, experience):
        self.memory.append(experience)

    # Sample a random batch of experiences from the replay buffer
    def sample(self):
        experiences = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*experiences)
        return np.vstack(states), actions, rewards, np.vstack(next_states), dones

    # Return the current size of the replay buffer
    def __len__(self):
        return len(self.memory)
