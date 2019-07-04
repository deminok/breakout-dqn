import gym
from gym import wrappers

from agent import *
from utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# mem = ReplayMemory()
# reward = 0
# action = 1
#
# def random_frame():
#     return np.random.randint(255, size=(84, 84))
#
#
# for i in range(RPL_MEM_SIZE):
#      mem.push(torch.tensor(random_frame(), dtype=torch.uint8), action, reward,
#               torch.tensor(random_frame(), dtype=torch.uint8))
#
# el = (mem.sample())
# print(getattr(el[1], 'action'))

# state = torch.zeros(size=(1,4,210,160), dtype=torch.float32)
# state_ = torch.ones(size=(1,4,210,160), dtype=torch.float32)
# observation = torch.randn(size=(210, 160), dtype=torch.float)
#
# state[:, :3, :, :] = state_[:, 1:, :, :]
#
# state[:, 3, :, :] = observation
# print(state)



env = gym.make('BreakoutDeterministic-v4')
agent = Agent(device)
frame_nr = 0

env = gym.wrappers.Monitor(env, './video/',video_callable=lambda episode_id: True,force = True)