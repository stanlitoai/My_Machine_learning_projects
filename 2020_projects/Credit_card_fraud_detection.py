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
print(tf.__version__)


data = pd.read_csv("creditcard.csv")
data.shape

data.isnull().sum()

data["Class"].value_counts()

#BALANCE DATASET
non_fraud = data[data["Class"]==0]
fraud = data[data["Class"]==1]

non_fraud.shape[0], fraud.shape[0]


non_fraud = non_fraud.sample(fraud.shape[0])

data = fraud.append(non_fraud, ignore_index=True)

X = data.drop("Class", axis=1)
y= data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 0, stratify=y)



scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)



y_train = y_train.to_numpy()
y_test = y_test.to_numpy()


X_train = X_train.reshape(787, 30, 1)
X_test = X_test.reshape(197, 30, 1)

#BUILD CNN

epochs=20

model = Sequential()
model.add(Conv1D(32, 2, activation = "relu", input_shape = X_train[0].shape))
model.add(BatchNormalization())
#model.add(MaxPool1D(2))
model.add(Dropout(0.2))


model.add(Conv1D(64, 2, activation = "relu"))
model.add(BatchNormalization())
#model.add(MaxPool1D(2))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64, activation = "relu"))
model.add(Dropout(0.5))


model.add(Dense(1, activation= "sigmoid"))

model.summary()

model.compile(optimizer= Adam(lr=0.0001), loss = "binary_crossentropy", metrics=["accuracy"])

history= model.fit(X_train, y_train, epochs = epochs, validation_data = (X_test, y_test), verbose=1)







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


plot_learningCurve(history, epochs)








































































