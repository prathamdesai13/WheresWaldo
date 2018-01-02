import matplotlib.pyplot as plt
import numpy as np

def normalize(pix):

    if pix < 0:

        return 0

    elif pix < 255:

        return pix

    else:

        return 255

def convolution(map, kernel, padding=1):

    padded_map = np.pad(map, (padding, padding), 'constant')
    k_h, k_w = kernel.shape
    map_h, map_w, _ = padded_map.shape
    map_c = map.shape[-1]

    conv_map = np.zeros((map_h - k_h + 2 * padding + 1, map_w - k_w + 2 * padding + 1, map_c))

    for channel in range(map_c):
        for height in range(map_h - k_h + padding):
            for width in range(map_w - k_w + padding):

                patch = padded_map[height:height + k_h, width:width + k_w, channel]

                conv_map[height, width, channel] = normalize(np.sum(np.multiply(patch, kernel)))

    return conv_map

def sharpen_kernel():

    return np.array([[0, -1, 0],
                    [-1, 5, -1],
                    [0, -1, 0]])


def outline_kernel():
    return np.array([[0, -1, 0],
                     [-1, 5, -1],
                     [0, -1, 0]])


if __name__ == '__main__':

    outline = outline_kernel()

    map = plt.imread("Maps/1.png")

    conv_map = convolution(map, outline)
    plt.imshow(map)
    plt.show()
    plt.imshow(conv_map)
    plt.show()
