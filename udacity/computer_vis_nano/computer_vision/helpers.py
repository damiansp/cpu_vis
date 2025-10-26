import glob
import os

import cv2
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


def standardize_input(image):  
    H = 600
    W = 1100
    standard_im = cv2.resize(image, (W, H))
    return standard_im


def encode(label):
    numerical_val = 1 if label == 'day' else 0
    return numerical_val


def standardize(image_list):
    standard_list = []
    for item in image_list:
        image, label = item
        standardized_im = standardize_input(image)
        binary_label = encode(label)    
        standard_list.append((standardized_im, binary_label))        
    return standard_list
