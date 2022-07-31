from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from Ipython.display import clear_output
#from six.moves import urllib
from sklearn.model_selection import train_test_split



import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf

CSV_COLUMN_NAMES = ["Sepallength", "Sepalwidth", "Petallength", "Petalwidth", "Species"]
SPECIES = ["Setosa", "Versicolor", "Virginica"]

datasets = pd.read_csv("iris.csv", names = CSV_COLUMN_NAMES, header = 0)

X = datasets
y = datasets.pop("Species")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train.keys()

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
        n_classes = 3
        )
x = lambda:input_fn(X_train, y_train, training = True)
x()

classifier.train(
        input_fn = x,
        steps = 5000
        )

eval_result = classifier.evaluate(
        input_fn = lambda: input_fn(X_test, y_test, training = False)
        )


print("\nTest set accuracy: {accuracy:0.3f}\n".format(**eval_result))

#Prediction

def input_fn(features, batch_size = 256):
    #Convert the input to a dataset without label
    return tf.data.Dataset.from_tensor_slices(dict(features)).batch(batch_size)

features = ["Sepallength", "Sepalwidth", "Petallength", "Petalwidth"]
predict = {}

print("Please type numeric  value as prompted")

for feature in features:
    valid= True
    while valid:
        val = input(feature + ":")
        if not val.isdigit():
            valid = False
            
    predict[feature]= [float(val)]
    
predictions = classifier.predict(input_fn = lambda: input_fn(predict))

for pred_dict in predictions:
    print("predictions", pred_dict)
    class_id = pred_dict["class_ids"][0]
    probability = pred_dict["probabilities"][class_id]
    
    print("prediction is '{}' ({:.1f}%)".format(
            SPECIES[class_id], 100 * probability))




#Here is some example input and expected classes you can try
    
expected = ["Setosa", "Versicolor", "Virginica"]

predict_x = {
        "Sepallenth" : [5.1, 5.9, 6.9],
        "Sepalwidth" : [3.3, 3.0, 3.1],
        "Petallength": [1.7, 4.2, 5.4],
        "Petalwidth" : [0.5, 1.5, 2.1],
        }













#using lambda
#x = lambda: print("hy")
#x()





















