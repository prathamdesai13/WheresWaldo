from PIL import Image
import os
from Heat.HeatMap import process as heat
import matplotlib.pyplot as plt
from scipy import misc

def jpg2png(images):

    for key in images:

        ims = images[key]
        for i, path in enumerate(ims):
            im = Image.open(key + '/' + path)
            im = heat(filepath=None, im=im)
            try:
                os.mkdir(key + 'png/')
            except OSError:

                pass

            im.save(key + 'png/' + str(i) + '.png')


def proc(images):
    i = 0
    for key in images:

        image = images[key]
        for path in image:

            img = misc.imread(key + '/' +path) / 255.0
            imagio = heat(filepath=None, im=img)
            misc.imsave('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128bwprocess/' + str(i) + '.png', imagio)
            i += 1

def collect_image_paths(path):

    image_paths = os.listdir(path)

    return image_paths

if __name__ == '__main__':

    images = dict()

    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/waldopng'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/waldopng')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/notwaldopng'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/notwaldopng')


    #jpg2png(images)

    proc(images)
