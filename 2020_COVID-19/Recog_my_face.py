import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from tqdm import tqdm
from tensorflow.keras.preprocessing import image

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split


pa = "blood_cell"
data = pd.read_csv("annotations.csv")

img_weight = 350
img_height = 350

X= []
for i in tqdm("chidera//"):
    print(i)
#    path = "chidera//"
    img = image.load_img(i, target_size = (img_weight, img_height, 3))
    img =image.img_to_array(img)
    img = img/255.0
    X.append(img)
    