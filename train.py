import datetime
from collections import deque
from copy import deepcopy

import gym

from agent import *
from eval import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

env = gym.make('BreakoutDeterministic-v4')
agent = Agent(device)

scores, mean_scores, episodes, frame_nr = deque(maxlen=EVAL_PERIOD), [], [], 0

session_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for i_episode in range(NUM_EPISODES):
    done = False
    score = 0

    #initiliaze Tensor to store states t and t+1
    states = torch.zeros(size=(1,5,84,84), dtype=torch.uint8)

    frame = env.reset()

    while not done:

        frame_nr += 1

        if RENDER:
            env.render()

        #Select action
        action = agent.select_action(frame_nr, states)

        #Send action to the emulator
        observation, reward, done, info = env.step(action)
        score += reward


        #Store experience in replay memory
        observation = psi(observation)
        """
            states[:, :4, :, :] returns four frames of state t
            states[:, 1:, :, :] returns four frames of state t+1
        """

        states[:, :4, :, :] = states[:, 1:, :, :]
        states[:, 4, :, :] = observation
        agent.memory.push(deepcopy(states), action, np.sign(reward), done)

        #Start training after replay memory has been populated with RPL_START_SIZE experiences
        if (frame_nr > RPL_START_SIZE) & (frame_nr % UPDATE_FREQUENCY == 0):
            agent.train()

            #update target network weights
            if frame_nr % NET_UPDATE_FREQ == 0:
                agent.update_target_net()

        #decrease epsilon
        if agent.epsilon > EPS_END:
            agent.epsilon = EPS_START - (0.9 / EPS_ANNEAL_STEPS) * frame_nr

    scores.append(score)

    if EVAL & (i_episode % EVAL_PERIOD == 0) & (i_episode != 0):
        episodes.append(i_episode)
        mean_scores.append(np.mean(scores))
        plot_reward(mean_scores, episodes, session_id)

    if LOGGING:
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              ": ",
              "episode: ", i_episode,
              " | score: ", score,
              " | epsilon: ", np.round(agent.epsilon, 2),
              " | replay memory length: ", agent.memory.__len__(),
              " | total time steps: ", frame_nr)
    env.close()





















