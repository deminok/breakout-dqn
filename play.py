import datetime
from collections import deque

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

    state = torch.zeros(size=(1,5,84,84), dtype=torch.uint8)

    if RECORD:
        env = gym.wrappers.Monitor(env, './videos/' + str(i_episode), video_callable=lambda episode_id: True, force=True)
    frame = env.reset()

    while not done:
        frame_nr += 1
        env.render()



        #Select action with highest Q value
        state = torch.tensor(np.float32(state[:, :4, :, :] / 255), dtype=torch.float32)
        action = agent.policy_net(state)
        action = int(action.max(1)[1])

        #Send action to the emulator
        observation, reward, done, info = env.step(action)
        score += reward

        #set state for next iteration
        observation = psi(observation)

        state[:, :3, :, :] = state[:, 1:, :, :]
        state[:, 3, :, :] = observation



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


















