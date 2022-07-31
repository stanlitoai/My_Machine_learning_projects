%tensorflow_version 2.x  # this line is not required unless you are in a notebook
from tensorflow.keras.preprocessing import sequence
import tensorflow as tf
import os
import numpy as np

np.std

path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

from google.colab import files
path_to_file = list(files.upload().keys())[0]


# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))


# Take a look at the first 250 characters in text
print(text[:250])































