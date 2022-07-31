import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, Flatten, Dense, MaxPool2D, Dropout, BatchNormalization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.optimizers import Adam


data = pd.read_csv("voice.csv")

data.isnull()

X = data.drop(labels=["label"], axis=1)
y=data["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state=0, stratify=y)

encoder = LabelEncoder()

y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

X_train[0].shape
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


X_train = X_train.reshape(2534, 20, 1)
X_test = X_test.reshape(634, 20, 1)

#Building your model

model = Sequential()
model.add(Conv1D(filters = 32, kernel_size = 2, activation = "relu",
                 input_shape =X_train[0].shape))
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

history = model.fit(X_train, y_train, epochs=60, validation_data=(X_test, y_test), verbose=1)



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


plot_learningCurve(history, 60)

classes_name= ["male", "female"]




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

#matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, figsize=(5,5), show_normed=True)
plt.tight_layout()
plt.savefig("Rg.png")
plt.show()



y_score = model.predict_proba(X_test)[:,1]

from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred)


from sklearn.metrics import accuracy_score

accuracy_score(y_test, y_pred)*100



y_pred = model.predict(X_test)















































