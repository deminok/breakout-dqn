import matplotlib.pyplot as plt

plt.style.use('ggplot')


def plot_reward(mean_scores, episodes, session_id):
    """Evaluation function to visualize training progress
    plots episodes against mean_scores and updates the graph in ./eval/
    until the training session is over

    :param
        mean_scores: array of mean games scores averaged over the last training trainig episodes
        episodes: array of numbers from zero to the current training episode
        session_id: unique identifier for each training session, used to save graph for each
        training session under a different name
    """
    plt.figure(2)
    plt.clf()
    plt.title('Average Reward on Breakout')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(episodes, mean_scores)
    plt.savefig("./eval/results " + session_id + ".png")
