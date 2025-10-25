import glob
import os

import matplotlib.image as mpimg


# This function loads in images and their labels and places them in a list
# The list contains all images and their associated labels
# For example, after data is loaded, im_list[0][:] will be the first image-label pair in the list
def load_dataset(image_dir):
    im_list = []
    image_types = ['day', 'night']
    for im_type in image_types:
        for f in glob.glob(os.path.join(image_dir, im_type, '*')):
            im = mpimg.imread(f)
            if im is not None:
                im_list.append((im, im_type))
    return im_list


