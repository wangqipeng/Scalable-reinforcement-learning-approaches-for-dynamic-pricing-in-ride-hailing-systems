import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = []
        self.capacity = capacity

    def push(self, state, action, reward, next_state):
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append((state, action, reward, next_state))

    def sample(self, batch_size):
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states = zip(*[self.buffer[i] for i in batch])
        return (torch.FloatTensor(states), torch.FloatTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(next_states))

    def __len__(self):
        return len(self.buffer)