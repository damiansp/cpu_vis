import os

import PIL
from   PIL import Image, ImageOps
from   tensorflow import keras
from   tensorflow.keras import Input, Model
from   tensorflow.keras.layers import (
    Activation, add, BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D,
    SeparableConv2D, UpSampling2D)
from   tensorflow.keras.preprocessing.image import load_img

from   prep import Batcher


IMG = '../../img'
INPUT_DIR = f'{IMG}/unet_images'
TARGET_DIR = f'{IMG}/unet_annotations/trimaps/'
IMG_SIZE = (160, 160)
N_CLASSES = 3
BATCH = 32


def main():
    input_img_paths, target_img_paths = prep_data()
    display_sample(input_img_paths[10], target_img_paths[10])
    keras.backend.clear_session()
    mod = get_mod()
    print(mod.summary())
    

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
        

def get_mod():
    inputs = Input(shape=IMG_SIZE + (3,))

    # First half of network: downsampling inputs
    # Entry block
    x = Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    prev_block_activation = x # set aside resid

    # Blocks 1, 2, 3 identical except for depth
    for filters in [64, 128, 256]:
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(3, strides=2, padding='same')(x)
        resid = Conv2D(
            filters, 1, strides=2, padding='same'
        )(prev_block_activation)
        x = add([x, resid]) # add resid back in
        prev_block_activation = x

    # Second half: upsampling inputs
    for filters in [256, 128, 64, 32]:
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2DTranspose(filters, 3, padding='same')(x)
        x = BatchNormalization()(x)
        x = UpSampling2D(2)(x)
        resid = UpSampling2D(2)(prev_block_activation)
        resid = Conv2D(filters, 1, padding='same')(resid)
        x = add([x, resid]) # back in
        prev_block_activation = x

    # Per-pixel classification
    outputs = Conv2D(N_CLASSES, 3, activation='softmax', padding='same')(x)
    mod = Model(inputs, outputs)
    return mod


if __name__ == '__main__':
    main()
