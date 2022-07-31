import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D, MaxPool1D
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
#from sklearn import datasets, metrrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

cancer = pd.read_csv("data.csv")

X = cancer.drop(columns=["target"], axis=1)
y = cancer.pop("target")





from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X["diagnosis"] = labelencoder.fit_transform(X["diagnosis"])



#onehotencoder = OneHotEncoder(handle_unknown='ignore' )
#X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0, stratify = y )


scaler = StandardScaler()

X_train.shape
X_test.shape

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = X_train.reshape(455, 32, 1)
X_test = X_test.reshape(114, 32, 1)

epochs = 60

model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2, activation = "relu", input_shape = (32, 1)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters = 64, kernel_size = 2, activation = "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))


model.summary()

model.compile(loss = "binary_crossentropy", optimizer=Adam(lr= 0.00005), metrics = ["accuracy"])

history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test),
                    verbose=1)




from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)

accuracy_score(y_test, y_pred)

pred = model.predict(X_test)




help(model)

history.history

#Plot training and validation accuracy values
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


plot_learningCurve(history, epochs)






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



























