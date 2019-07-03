import torch
import numpy as np

from config import *
from replay_memory import *

mem = ReplayMemory()
reward = 0
action = 1

def random_frame():
    return np.random.randint(255, size=(4, 84, 84))


for i in range(RPL_MEM_SIZE):
     mem.push(torch.tensor(random_frame(), dtype=torch.uint8), action, reward,
              torch.tensor(random_frame(), dtype=torch.uint8))

el = (mem.sample())
print(getattr(el[1], 'action'))