import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
print(tf.__version__)
#%matplotlib


mnist = keras.datasets.fashion_mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data()

class_names = ["top","trouser", "pullover", "dress", "coat", "sandal", "shirt", "sneaker", "bag", "ankle boot"]

plt.figure()
plt.imshow(X_train[0])
plt.colorbar()
plt.show()

X_train = X_train / 255.0
X_test = X_test / 255.0

#building the model

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation = "relu"))
model.add(Dense(10, activation = "softmax"))

model.summary()

#Compilation

model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])

history = model.fit(X_train, y_train, epochs = 10, batch_size=100, validation_split=0.2)

test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_acc)


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)

accuracy_score(y_test, y_pred)

pred = model.predict(X_test)
np.argmax(pred[504])
""""
plt.figure()
plt.imshow(X_train[504])
#plt.imshow(np.argmax(pred[0]))
plt.colorbar()
plt.show()
""""
%matplotlib

help(model)

history.history

#Plot training and validation accuracy values
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model accuracy")
plt.ylabel("Accuracy")
plt.grid()
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc= "upper left")
plt.show()



#Plot training and validation loss values
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model loss")
plt.grid()
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Val"], loc= "upper left")
plt.show()



#Plot confusion matrix
font = {
        "family": "Times New Roman",
        "weight": "bold",
        "size": 14
        
        }


import matplotlib
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#to reset matplotli
matplotlib.rcParams.update(matplotlib.rcParamsDefault)

matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, figsize=(6,6), class_names= class_names, show_normed=True)
plt.tight_layout()
plt.savefig("cm.png")
plt.show()























