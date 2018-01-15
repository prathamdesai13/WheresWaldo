import numpy as np
import colorsys
import matplotlib.pyplot as plt

def rgb_to_primary(map):

    primary_map = np.zeros_like(map)

    for h in range(map.shape[0]):
        for w in range(map.shape[1]):

            pixel = map[h, w]

            for i in range(len(pixel)):

                if pixel[i] > 127.0 / 255.0:

                    pixel[i] = 1.0

                else:

                    pixel[i] = 0.0

            primary_map[h, w] = pixel

    return primary_map

def assign_color(pixel, color):

    for c in color:

        if pixel[0] == c[0] and pixel[1] == c[1] and pixel[2] == c[2]:

            return True

    return False


def filter(map, color):

    mono_map = np.zeros_like(map)

    for h in range(map.shape[0]):
        for w in range(map.shape[1]):

            pixel = map[h, w]

            if assign_color(pixel, color):

                mono_map[h, w] = pixel

            else:

                mono_map[h, w] = (0.0, 1.0, 0.0)

    return mono_map

def get_waldo_stripes(map):

    stripes = np.zeros_like(map)

    for height in range(map.shape[0]):
        for width in range(map.shape[1]):

            r, g, b = map[height, width]

            h, s, v = colorsys.rgb_to_hsv(r, g, b)

            if (h < 0.05 or h > 0.95) and s > 0.4 and v > 0.4:

                stripes[height, width] = map[height, width]

            elif s < 0.2 and v > 0.9:

                stripes[height, width] = map[height, width]

    return stripes

def process(filepath):

    white = (1.0, 1.0, 1.0)
    red = (1.0, 0.0, 0.0)

    map = plt.imread(filepath)
    primary_map = rgb_to_primary(map)
    filter_map = filter(primary_map, color=[red, white])
    stripes = get_waldo_stripes(filter_map)

    return stripes
