from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from Ipython.display import clear_output
#from six.moves import urllib
from sklearn.model_selection import train_test_split

import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
#%matplotlib

CSV_COLUMN_NAMES = ['Area', 'Garage', 'FirePlace', 'Baths', 'White Marble',"Black Marble", 'Indian Marble',
       'Floors', 'City', 'Solar', 'Electric', 'Fiber', 'Glass Doors',
       'Swiming Pool', 'Garden', "Prices"]
datasets = pd.read_csv("HousePrices.csv")

df = datasets.copy()
df.tail()

#cleaning
df.isna().sum()

#dropping those row
df= df.dropna()
df.sample(frac=0.8, random_state=0)
#get the dummies

df = pd.get_dummies(df, prefix="", prefix_sep = "")
df.tail()
#datasets.drop(["Black Marbel"])

X = datasets
y = datasets.pop("Prices")
z = datasets.pop("Black Marble")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

train_data = tf.data.experimental.make_csv_dataset(
        train_dataset_fp,
        batch_
        )



































def input_fn(features, labels, training = True, batch_size = 256):
    #Convert theinput to a dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    
    #shuffle and  repect if you are in training mode.
    if training:
        dataset = dataset.shuffle(1000).repeat()
        
    return dataset.batch(batch_size)


my_feature_columns = []

for key in X_train.keys():
    
    my_feature_columns.append(tf.feature_column.numeric_column(key=key))
    
print(my_feature_columns)


#Building a DNN with 2 hidden layers with 30 and 10 hidden nodes each

classifier = tf.estimator.DNNClassifier(
        feature_columns =my_feature_columns, 
        #two hidden layers of 30 and 10 nodes respectively
        hidden_units = [30, 10],
        #The model must choose between 3 classes
        n_classes = 2
        )

x = lambda:input_fn(X_train, y_train, training = True)
x()

classifier.train(
        input_fn =x ,
        steps = 5000
        )

eval_result = classifier.evaluate(
        input_fn = lambda: input_fn(X_test, y_test, training = False)
        )














