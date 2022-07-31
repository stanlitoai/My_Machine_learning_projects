#Build 2D CNN

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import tensorflow as tf


from  tensorflow.keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

classes_name = [0, 1, 2, 3, 4, 5, 6, 7,8 , 9]

X_train = X_train / 255.0
X_test = X_test / 255.0

X_train.shape


X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

input_shape = X_train[0].shape

#Building
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation = "relu", input_shape = input_shape))
model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))


model.add(Flatten())
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))


model.summary()


model.compile(loss= "sparse_categorical_crossentropy", optimizer = "adam", metrics =["accuracy"])

history = model.fit(X_train, y_train, batch_size = 128, epochs = 10, verbose = 1, 
                    validation_data =(X_test, y_test))


model.evaluate(X_test, y_test)


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)

accuracy_score(y_test, y_pred)

pred = model.predict(X_test)








def plot_learningCurve(history, epoch):
    
    #Plot training and validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history["accuracy"])
    plt.plot(epoch_range, history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()
    
    
    
    #Plot training and validation loss values
    plt.plot(epoch_range, history.history["loss"])
    plt.plot(epoch_range, history.history["val_loss"])
    plt.title("Model loss")
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()


plot_learningCurve(history, 10)







import matplotlib
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#to reset matplotli
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, figsize=(6,6), class_names= classes_name, show_normed=True)
plt.tight_layout()
plt.savefig("Rgs.png")
plt.show()





















































