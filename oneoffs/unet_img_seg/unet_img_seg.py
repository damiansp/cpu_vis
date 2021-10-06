import os

import PIL
from   PIL import Image, ImageOps
from   tensorflow.keras.preprocessing.image import load_img


IMG = '../../img'
INPUT_DIR = f'{IMG}/unet_images'
TARGET_DIR = f'{IMG}/unet_annotations/trimaps/'
IMG_SIZE = (160, 160)
N_CLASSES = 3
BATCH = 32


def main():
    input_img_paths, target_img_paths = prep_data()
    display_sample(input_img_paths[10], target_img_paths[10]))
    

    

def prep_data():
    input_img_paths = sorted([os.path.join(INPUT_DIR, f)
                              for f in os.listdir(INPUT_DIR)
                              if f.endswith('.jpg')])
    target_img_paths = sorted([os.path.join(TARGET_DIR, f)
                               for f in os.listdir(TARGET_DIR)
                               if f.endswith('.png') and not f.startswith('.')])
    print('n samples:', len(input_img_paths))
    for i_path, t_path in zip(input_img_paths[:10], target_img_paths[:10]):
        print(i_path, '|', t_path)
    return input_img_paths, target_img_paths


def display_sample(in_path, target_path):
    img = Image.open(in_path)
    img.show()
    img = ImageOps.autocontrast(load_img(target_path))
    img.show()
        

if __name__ == '__main__':
    main()
