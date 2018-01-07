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

def pixel_by_pixel(map, colours):

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


def rgb_to_gray(map):
    # rgbtogray mapping: 0.299R + 0.587G + 0.114B
    gray_map = np.dot(map, np.array([0.299, 0.587, 0.114]))

    return gray_map


def rgb_to_primary(map):

    primary_map = np.zeros_like(map)

    for h in range(map.shape[0]):
        for w in range(map.shape[1]):

            pixel = map[h, w]

            r, g, b = pixel

            if r > 127.0 / 255.0:

                r = 1.0

            else:

                r = 0.0

            if g > 127.0 / 255.0:

                g = 1.0

            else:

                g = 0.0

            if b > 127.0 / 255.0:

                b = 1.0

            else:

                b = 0.0

            primary_map[h, w] = (r, g, b)

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

    return mono_map

def crop_map(map, crop):

    cropped = np.zeros_like(map)
    crop_height = crop.shape[0]
    crop_width = crop.shape[1]

    print(crop.shape)
    for h in range(int(map.shape[0] / crop_height)):
        for w in range(int(map.shape[1] / crop_width)):

            square = map[crop_height * h:crop_height * (h + 1), crop_width * w:crop_width * (w + 1)]

            error = abs(np.sum(square - crop)) / (crop_height * crop_width)
            print(error)
            if error < 0.3:

                cropped[crop_height * h:crop_height * (h + 1), crop_width * w:crop_width * (w + 1)] = square

            else:

                cropped[crop_height * h:crop_height * (h + 1), crop_width * w:crop_width * (w + 1)] = (0.0, 0.0, 0.0)

    return cropped


if __name__ == '__main__':

    white = (1.0, 1.0, 1.0)
    black = (0.0, 0.0, 0.0)
    red = (1.0, 0.0, 0.0)

    map = plt.imread("Maps/9.png")

    #heatmap = pixel_by_pixel(map, [(0.8666666666666667, 0.09411764705882353, 0.10980392156862745)])

    grey_map = rgb_to_gray(map)
    primary_map = rgb_to_primary(map)
    filter_map = filter(primary_map, color=[(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)])

    crop = np.zeros((30, 30, 3))

    for h in range(30):

        if h < 6:

            crop[h, :, :] = red

        elif 6 < h < 16:

            crop[h, :, :] = white

        elif 16 < h < 24:

            crop[h, :, :] = red

        else:

            crop[h, :, :] = black

    cropped = crop_map(filter_map, crop)
    plt.imshow(map)
    plt.show()

    plt.imshow(grey_map, plt.get_cmap('gray'))
    plt.show()

    plt.imshow(primary_map)
    plt.show()

    plt.imshow(filter_map)
    plt.show()

    plt.imshow(cropped)
    plt.show()
