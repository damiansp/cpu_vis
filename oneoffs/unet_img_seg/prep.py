import numpy as np
from tensorflow.keras.utils.Sequence
from tensorflow.keras.preprocessing import load_img


class Batcher(Sequence):
    '''Helper to iterate over data as np.arrays'''
    def __init__(self, batch_size, img_size, input_paths, target_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_paths = input_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.target_paths) // self.batch_size

    def __getitem__(self, idx):
        '''Return tuple (input, target) at batch <idx>'''
        i = idx * self.batch_size
        batch_input_paths = self.input_paths[i:i + self.batch_size]
        batch_target_paths = self.target_paths[i:i + self.batch_size]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype='float32')
        for j, path in enumerate(batch_input_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype='uint8')
        for j, path in enumerate(batch_target_paths):
            img = load_img(
                path, target_size=self.img_size, color_mode='grayscale')
            y[j] = np.expand_dims(img, 2)
            # Ground truth labels are 1, 2, 3; subtract one to get 0, 1, 2
            y[j] -= 1
        return x, y
