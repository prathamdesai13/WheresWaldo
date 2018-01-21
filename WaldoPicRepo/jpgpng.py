from PIL import Image
import os
from Heat.HeatMap import process as heat
from scipy import misc

# this code is very poorly written, i would not advising using it for anythign

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
