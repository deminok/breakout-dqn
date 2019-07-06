import numpy as np
import torch.optim.rmsprop as optim

from dqn import *
from replay_memory import *
from utils import *


class Agent():
    def __init__(self, device):
        self.epsilon = EPS_START

        #set up policy and target DQN, optimizer and replay memory
        self.policy_net = DQN().to(device)
        self.target_net = DQN().to(device)
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LEARNING_RATE,
                                       momentum=GRAD_MOM, eps=MIN_SQ_GRAD)
        self.memory = ReplayMemory()

        # initialize target net
        self.update_target_net()

    def select_action(self, frame_nr, states): #0 NOOP, 1 fire, 2 right, 3 left
        """
        Selects an action based on epsilon greedy strategy
        :param
            frame_nr: current training frame, used to make sure that only random actions are selected
            in the beginning
            states: pytorch Tensor of size (1, 5, 84, 84), type torch.uint8 of which the first four
            frames are forwarded through the policy net to calculate Q values for each possible action
        :return: max Q value, integer indicating which action the agent should take: 0 NOOP, 1 fire, 2 right, 3 left
        """
        if (frame_nr < RPL_START_SIZE) | (np.random.rand() <= self.epsilon):
            action = random.randrange(ACTION_SIZE)

        else:
            state = torch.tensor(np.float32(states[:, :4, :, :] / 255), dtype=torch.float32)
            action = self.policy_net(state)
            action = int(action.max(1)[1])
        return action

    def update_target_net(self):
        """Copies weights of policy net to target net"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def train(self):
        """
        Function to update parameters of the policy net
        Samples minibatch and performs update iteration according to the algorithm
        given by Mnih. et al. (2015)
        """
        mini_batch = self.memory.sample()
        states, actions, rewards = [], [], []
        done = 0
        input, target = [], []

        for i in range(BATCH_SIZE):
            states.append(np.float32(getattr(mini_batch[i], 'states')) / 255.)
            actions.append(getattr(mini_batch[i], 'action'))
            rewards.append(getattr(mini_batch[i], 'reward'))
            done = getattr(mini_batch[i], 'done')

        states = torch.tensor(states, dtype=torch.float32)

        # Calculate Q(s_j, a_j, theta) for each element in the minibatch
        for j in range(BATCH_SIZE):
            Q_j = self.policy_net(states[j][:, :4, :, :])[0][actions[j]]
            input.append(Q_j)

        input = torch.stack(input)

        # Calculate r_j + GAMMA * max Q(s_j, a_j, theta_) for each element in the minibatch
        for j in range(BATCH_SIZE):
            if done == 1:
                y_j = torch.tensor(rewards[0], dtype=torch.float32)
                target.append(y_j)

            else:
                y_j = rewards[0] + GAMMA * self.target_net(states[j][:, 1:, :, :])[0].max(0)[0]
                target.append(y_j)

        target = torch.stack(target)

        #loss function
        loss = F.mse_loss(input, target)
        loss.backward()

        #perform one update on policy_net
        self.optimizer.step()
