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
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold


data = pd.read_csv("santander-train.csv")
data.shape
X = data.drop(labels = ["TARGET","ID"], axis=1)
y= data["TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state=0, stratify =y)

##Removing constant , some constant and DUplicate features

filter= VarianceThreshold(0.01)

X_train = filter.fit_transform(X_train)
X_test = filter.transform(X_test)


X_train_T = X_train.T
X_test_T = X_test.T


X_train_T = pd.DataFrame(X_train_T)
X_test_T = pd.DataFrame(X_test_T)


X_train_T.duplicated().sum()

duplicated_features = X_train_T.duplicated()

features_to_keep = [not index for index in duplicated_features]

X_train = X_train_T[features_to_keep].T

X_train.shape

X_test = X_test_T[features_to_keep].T

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = X_train.reshape(60816, 256, 1)
X_test = X_test.reshape(15204, 256, 1)



y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#BUILD CNN

model = Sequential()
model.add(Conv1D(32, 3, activation = "relu", input_shape = (256, 1)))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.3))

model.add(Conv1D(64, 3, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Conv1D(128, 3, activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool1D(2))
model.add(Dropout(0.5))


model.add(Flatten())

model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))

model.add(Dense(1, activation = "sigmoid"))


model.summary()

model.compile(optimizer = Adam(lr=0.00005), loss= "binary_crossentropy", metrics = ["accuracy"])


history = model.fit(X_train, y_train, epochs= 10, validation_data = (X_test, y_test), verbose =1 )



test_loss, test_acc = model.evaluate(X_test, y_test)

print(test_acc)


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_test)

accuracy_score(y_test, y_pred)




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


plot_learningCurve(history, 10)



























































