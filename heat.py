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

def epislon_range(pixel, c):

    epsilon = 5.0 / 255.0
    if c - epsilon < pixel < c + epsilon:

        return pixel

    else:

        return 0.0

def heat_map(map, colours):

    heat = np.zeros_like(map)

    for c in colours:
        r, g, b = c

        for channel in range(map.shape[-1]):
            for height in range(map.shape[0]):
                for width in range(map.shape[1]):

                    pixel = map[height, width, channel]

                    if channel == 0:

                        heat[height, width, channel] = epislon_range(pixel, r)

                    elif channel == 1:

                        heat[height, width, channel] = epislon_range(pixel, g)

                    elif channel == 2:

                        heat[height, width, channel] = epislon_range(pixel, b)


    return heat

if __name__ == '__main__':

    map = plt.imread("Maps/9.png")

    heatmap = heat_map(map, [(0.8666666666666667, 0.09411764705882353, 0.10980392156862745)])

    plt.imshow(map)
    plt.show()

    plt.imshow(heatmap)
    plt.show()
