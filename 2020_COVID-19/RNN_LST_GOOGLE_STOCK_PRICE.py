import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv("Google.csv", date_parser = True)

data.head()
data.tail()


data_train = data[data["Date"]< "2020-01-01"].copy()
data_test = data[data["Date"]>= "2020-01-01"].copy()


X= data_train.drop(["Date", "Adj Close"], axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

X_train = []
y_train = []

for i in range(60, X.shape[0]):
    X_train.append(X[i-60, i])
    y_train.append(X[i, 0])

























































































