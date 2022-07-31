import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPool2D, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

(X_train,y_train), (X_test, y_test) = cifar10.load_data()

classes_name = ["airplane", "automobile", "birds", "cat", "deer", "dog",
                "frog", "horse", "ship", "truck"]



#X_train = X_train/255.0
#X_test = X_test/255.0

X_train = X_train / 255.0
X_test = X_test / 255.0

print(len(X_train))
print(len(y_train))

print(len(X_test))
print(len(y_test))


plt.imshow(X_train[100])
plt.show()

X_train[100].shape

#Build CNN MOdel
model = Sequential()
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding="same",
                 activation="relu", input_shape= (32, 32, 3)))
model.add(Conv2D(filters = 32, kernel_size =(3,3), padding="same", 
                 activation="relu"))
model.add(MaxPool2D(pool_size = (2,2), strides=2, padding="valid"))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units =10, activation="softmax"))


model.summary()


model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ["sparse_categorical_accuracy"])

history = model.fit(X_train, y_train, batch_size=10, epochs= 10,
                    verbose=1, validation_data=(X_test, y_test))


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)

accuracy_score(y_test, y_pred)

pred = model.predict(X_test)




help(model)

history.history

#Plot training and validation accuracy values
epoch_range = range(1, 11)
plt.plot(epoch_range, history.history["sparse_categorical_accuracy"])
plt.plot(epoch_range, history.history["val_sparse_categorical_accuracy"])
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
fig, ax = plot_confusion_matrix(conf_mat=mat, figsize=(6,6), class_names= classes_name, show_normed=True)
plt.tight_layout()
plt.savefig("Rg.png")
plt.show()





























