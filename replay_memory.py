import random
from collections import namedtuple

from config import *

Experience = namedtuple('Experience',
                        ('state', 'action', 'reward', 'state_'))


class ReplayMemory(object):

    def __init__(self):
        self.capacity = RPL_MEM_SIZE
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, state, action, reward, state_):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, state_)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)
