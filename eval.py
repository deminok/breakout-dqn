import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plot_reward(mean_scores, episodes, session_id):
    plt.figure(2)
    plt.clf()
    plt.title('Average Reward on Breakout')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episodes, mean_scores)
    plt.savefig("./eval/results " + session_id + ".png")
    #plt.show()
