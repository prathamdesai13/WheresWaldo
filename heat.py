import matplotlib.pyplot as plt
import numpy as np

def convolution(map, kernel, padding=1):

    padded_map = np.pad(map, (padding, padding), 'constant')
    k_h, k_w = kernel.shape
    map_h, map_w, _ = padded_map.shape
    map_c = map.shape[-1]

    conv_map = np.zeros(map.shape)

    for height in range(map_h - k_h + 1):
        for width in range(map_w - k_w + 1):
            for channel in range(map_c):

                patch = padded_map[height:height + k_h, width:width + k_w, channel]

                conv_map[height, width, channel] = np.sum(patch * kernel)

    return conv_map


if __name__ == '__main__':
    sharpen = np.array([[0, -1, 0],
                        [-1, 5, -1],
                        [0, -1, 0]])

    identity = np.array([[1, 0,0],
                         [0,1,0],
                         [0,0,1]])

    random_arr = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])
    map = plt.imread("Maps/2.png")

    conv_map = convolution(map, identity)
    print(conv_map.shape)
    #conv_map = np.multiply(sharpen, random_arr)
    #fig = plt.figure()
    #ax1 = fig.add_subplot(1, 2, 1)
    #ax1.imshow(map)
    #ax2 = fig.add_subplot(1, 2, 2)
    #ax2.imshow(conv_map)
    plt.imshow(map)
    plt.show()
    plt.imshow(conv_map)
    plt.show()

