import matplotlib.pyplot as plt
import torch
from skimage.color import rgb2gray
from skimage.transform import resize


def psi(screen):
    """Preprocessing function psi

    :param
        screen: 3x210x160 np.array, with RGB channels

    :return
        screen: 84x84 np.uint8, grayscale with luminance values [0, 255]
        cropped to the playing field of the game
    """
    screen = screen[32:193, 8:152]
    screen = resize(screen, (84, 84))
    screen = rgb2gray(screen)
    screen = torch.tensor(screen * 255)

    return screen


def plot_states(state):
    """Helper function to visualize states used for training
    Plots five consecutive game frames of state t (first four frames)
    and state t+1 (last four frames)

    :param
        state: pytorch tensor with size (1,5,84,84) of type torch.uint8
    """
    w = 160
    h = 210
    fig = plt.figure(figsize=(10, 4))
    columns = 4
    rows = 1
    for i in range(1, columns * rows + 1):
        img = state[0, i - 1, :, :]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img)
    plt.show()
