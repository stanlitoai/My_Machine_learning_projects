import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool1D, Conv1D
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold






data1 = pd.read_csv("train_customers.csv")
data2 = pd.read_csv("train_locations.csv")
data3 = pd.read_csv("SampleSubmission.csv")
data4 = pd.read_csv("orders.csv")
data5 = pd.read_csv("vendors.csv")


data1.shape
data2.shape
data3.shape
data4.shape
data5.shape

data3["target"].value_counts()


data1.isnull().sum()
data2.isnull().sum()
data3.isnull().sum()
data4.isnull().sum()
data5.isnull().sum()

data1.dropna(inplace = True)
data2.dropna(inplace = True)
data3.dropna(inplace = True)
data4.dropna(inplace = True)
data5.dropna(inplace = True)







































