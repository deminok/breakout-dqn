import datetime
from collections import deque
from copy import deepcopy

import gym
from gym import wrappers

from agent import *
from eval import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('BreakoutDeterministic-v4')
agent = Agent(device)
scores, mean_scores, episodes = deque(maxlen=EVAL_PERIOD), [], []
frame_nr = 0

session_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

for i_episode in range(NUM_EPISODES):
    done = False
    score = 0

    states = torch.zeros(size=(1,5,84,84), dtype=torch.uint8)

    if RECORD:
        env = gym.wrappers.Monitor(env, './videos/' + session_id + "/" + str(i_episode), video_callable=lambda episode_id: episode_id % 10 == 0)
    frame = env.reset()
    #print(state)
    while not done:

        frame_nr += 1

        if RENDER:
            env.render()

        #Select action
        action = agent.select_action(frame_nr, states)

        #Send action to the emulator
        observation, reward, done, info = env.step(action)
        score += reward
        """
        states[:, :4, :, :] => four frames of state t
        states[:, 1:, :, :] => four frames of state t+1
        """

        #Store experience in replay memory
        observation = psi(observation)

        states[:, :4, :, :] = states[:, 1:, :, :]
        states[:, 4, :, :] = observation
        agent.memory.push(deepcopy(states), action, np.sign(reward), done)

        #Start training after replay memory has been populated with RPL_START_SIZE experiences
        if (frame_nr > RPL_START_SIZE) & (frame_nr % UPDATE_FREQUENCY == 0):
            agent.train()

            #update target network weights
            if frame_nr % (UPDATE_FREQUENCY * NET_UPDATE_FREQ) == 0:
                agent.update_target_net()
                print('target net updated at frame ', frame_nr)

        #decrease epsilon
        if agent.epsilon > EPS_END:
            agent.epsilon = EPS_START - (0.9 / EPS_ANNEAL_STEPS) * frame_nr

    scores.append(score)

    if EVAL & (i_episode % EVAL_PERIOD == 0) & (i_episode != 0):
        episodes.append(i_episode)
        mean_scores.append(np.mean(scores))
        plot_reward(mean_scores, episodes, session_id)

    if LOGGING: #& (i_episode % LOGGING_PERIOD == 0) & (i_episode != 0):
        print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              ": ",
              "episode: ", i_episode,
              " | score: ", score,
              " | epsilon: ", np.round(agent.epsilon, 2),
              " | replay memory length: ", agent.memory.__len__(),
              " | total time steps: ", frame_nr)
    env.close()

if SAVE:
    torch.save(agent.policy_net, "./saved_model/breakout_dqn")



















