import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split


data = pd.read_csv("Churn_Modelling.csv")
data.head()

X = data.drop(labels =["Exited", "CustomerId","Surname","RowNumber"], axis =1)
y = data["Exited"]


from sklearn.preprocessing import LabelEncoder, StandardScaler

label = LabelEncoder()
X["Geography"] = label.fit_transform(X["Geography"])

X["Gender"] = label.fit_transform(X["Gender"])

X = pd.get_dummies(X, drop_first=True, columns=["Geography"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .2, random_state =0, stratify = y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


#Building the ANN
model = Sequential()
model.add(Dense(X.shape[1], activation="relu", input_dim = X.shape[1]))
model.add(Dense(128, activation = "relu"))
model.add(Dense(1, activation = "sigmoid"))


#compilation

model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

history= model.fit(X_train, y_train.to_numpy(), batch_size = 10, epochs= 20, verbose= 1 )

y_pred = model.predict_classes(X_test)

#pre = pd.DataFrame({"Actual ": y_test, "Prediction": y_pred})

model.evaluate(X_test, y_test.to_numpy())


from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, y_pred)

accuracy_score(y_test, y_pred)


#PLOTTING WITH CONFUSION MATRIC


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

matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=mat, figsize=(10,10), class_names= class_names, show_normed=True)

















