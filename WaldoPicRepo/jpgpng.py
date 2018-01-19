from PIL import Image
import os

def jpg2png(images):

    for key in images:

        ims = images[key]
        for i, path in enumerate(ims):
            im = Image.open(key + '/' + path)
            try:
                os.mkdir(key + 'png/')
            except OSError:

                pass

            im.save(key + 'png/' + str(i) + '.png')


def collect_image_paths(path):

    image_paths = os.listdir(path)

    return image_paths

if __name__ == '__main__':

    images = dict()

    """images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128/notwaldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-bw/notwaldo')"""
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-gray/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-gray/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-gray/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/128-gray/notwaldo')

    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256/notwaldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-bw/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-bw/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-bw/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-bw/notwaldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-gray/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-gray/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-gray/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/256-gray/notwaldo')

    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64/notwaldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-bw/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-bw/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-bw/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-bw/notwaldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-gray/waldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-gray/waldo')
    images['/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-gray/notwaldo'] = \
        collect_image_paths('/Users/niravdesai/Desktop/WheresWaldo/WaldoPicRepo/64-gray/notwaldo')

    jpg2png(images)
