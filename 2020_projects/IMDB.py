import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#from tensorflow.keras.datasets import imdb
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences



(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words= 20000)

X_train = pad_sequences(X_train, maxlen=100)
X_test = pad_sequences(X_test, maxlen=100)



































































