
import tensorflow as tf
from tensorflow import keras
import numpy as np 





class dataGen(tf.keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, x):
        self.batch_size  = batch_size
        self.input_imgs  = x
        #self.target_imgs = y

    def __len__(self):
        return self.input_imgs.shape[3] // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        
        i = idx * self.batch_size
        batch_input_imgs  = self.input_imgs[:,:,:, i : i + self.batch_size]
      #  print(batch_input_imgs.shape)
        x_set = np.transpose(batch_input_imgs, (3, 0, 1, 2))
        
        #print(x_set.shape)
       # print(x_set.shape)
        return x_set     