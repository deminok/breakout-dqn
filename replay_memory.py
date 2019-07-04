import random
from collections import namedtuple

from config import *

Experience = namedtuple('Experience',
                        ('states', 'action', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self):
        self.capacity = RPL_MEM_SIZE
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, states, action, reward, done):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(states, action, reward, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self):
        return random.sample(self.memory, BATCH_SIZE)
