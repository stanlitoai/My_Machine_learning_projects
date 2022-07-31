import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv1D, GlobalMaxPooling1D, Embedding
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



file_names = ['./data/category/Electronics_small.json', './data/category/Books_small.json',
              './data/category/Clothing_small.json', './data/category/Grocery_small.json',
              './data/category/Patio_small.json']

df_list = []
for file in file_names:
    df = pd.read_json(file, names = ["comment", "sentiment"]  )
    df_list.append(df)


trainingdata = pd.concat(df_list)


print("Number of rows: %d" % len(trainingdata))

trainingdata.head(10)
















