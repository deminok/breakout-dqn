import matplotlib.pyplot as plt
import torch
from skimage.color import rgb2gray
from skimage.transform import resize


def psi(screen):
    """Preprocessing function psi

    Args:
        screen: 3x210x160 np.array, with RGB channels

    Returns:
        screen: 84x84 np.uint8, grayscale luminance values [0, 255]
        cropped to the playing field of the game
    """
    screen = screen[32:193, 8:152]
    screen = resize(screen, (84, 84))
    screen = rgb2gray(screen)
    screen = torch.tensor(screen * 255)

    return screen

    #for testing
    # screen = np.uint8(mpimg.imread('screen.png') * 255)
    # plt.imshow(img, cmap='gist_gray')
    # plt.show()


def plot_states(state):
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


