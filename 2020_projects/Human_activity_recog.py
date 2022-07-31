import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
print(tf.__version__)

#load data
file = open("WISDM_ar_v1.1_raw.txt")
lines = file.readlines()
processedlist =[]
for i, line in enumerate(lines):
    try:
        line = line.split(",")
        last = line[5].split(";")[0]
        last = last.strip()
        if last == "":
            break;
        temp = [line[0], line[1], line[2], line[3], line[4], last]
        processedlist.append(temp)
        
    except:
        print("Error at line number: ", i)

columns = ["user", "activity", "time", "x", "y", "z"]

data = pd.DataFrame(data = processedlist, columns = columns)
data.head()
data.shape

data.info()
data.isnull().sum()

data["activity"].value_counts()

##BALANCE THIS DATA


data["x"] = data["x"].astype("float")
data["y"] = data["y"].astype("float")
data["z"] = data["z"].astype("float")


fs = 20
activities = data["activity"].value_counts().index


##VISUALIZATION

def plot_activity(activity, data):
    fig, (ax0, ax1, ax2) = plt.subplots(nrows =3, figsize=(11, 7), sharex=True)
    plot_axis(ax0, data["time"], data["x"], "X-Axis")
    plot_axis(ax1, data["time"], data["y"], "Y-Axis")
    plot_axis(ax2, data["time"], data["z"], "Z-Axis")
    plt.subplots_adjust(hspace= 0.2)
    fig.suptitle(activity)
    plt.subplots_adjust(top=0.90)
    plt.show()
    
    
    
def plot_axis(ax, x, y, title):
    ax.plot(x, y, "g")
    ax.set_title(title)
    ax.xaxis.set_visible(False)
    ax.set_ylim([min(y) - np.std(y), max(y) * np.std(y)])
    ax.set_xlim([min(x), max(x)])
    ax.grid(True)
    
    
    
for activity in activities:
    data_for_plot = data[(data["activity"] == activity)][:fs*10]
    plot_activity(activity, data_for_plot)
    

df = data.drop(["user", "time"], axis =1).copy()


data["activity"].value_counts()


Walking = df[df["activity"] == "Walking"].head(3555).copy()
Jogging = df[df["activity"] == "Jogging"].head(3555).copy()
Upstairs = df[df["activity"] == "Upstairs"].head(3555).copy()
Downstairs = df[df["activity"] == "Downstairs"].head(3555).copy()
Sitting = df[df["activity"] == "Sitting"].head(3555).copy()
Standing = df[df["activity"] == "Standing"].copy()


##BALANCED DATA
balanced_data = pd.DataFrame()
balanced_data = balanced_data.append([Walking, Jogging, Upstairs, Downstairs,
                                      Sitting, Standing])
    
balanced_data["activity"].value_counts()



#APPLYING LABELENCODER

from sklearn.preprocessing import LabelEncoder

label = LabelEncoder()


balanced_data["label"] = label.fit_transform(balanced_data["activity"])

label.classes_

##Standardized data

X = balanced_data[["x", "y", "z"]]

y = balanced_data["label"]

scaler = StandardScaler()

X = scaler.fit_transform(X)

scaled_x = pd.DataFrame(data = X, columns = ["x", "y", "z"])

scaled_x["label"] = y.values


##FRAME PREPARATION
import scipy.stats as stats


fs = 20
frame_size = fs*4
hop_size = fs*2


def get_frame(df, frame_size, hop_size):
    
    N_FEATURES = 3
    frames = []
    labels = []
    
    for i in range(0, len(df) - frame_size, hop_size):
        x = df["x"].values[i: i + frame_size]
        y = df["y"].values[i: i + frame_size]
        z = df["z"].values[i: i + frame_size]
        
        
        ##Retrieve the most often used label in this segment
        
        label = stats.mode(df["label"][i: i + frame_size])[0][0]
        frames.append([x, y, z])
        labels.append([label])
        
    ##Bring the segments into a better shape
    frames = np.asarray([frames]).reshape(-1, frame_size, N_FEATURES)
    labels = np.asarray(labels)
    
    return frames, labels
        
X, y = get_frame(scaled_x, frame_size, hop_size)


X.shape
y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2,
                                                    random_state = 0, stratify = y)


X_train.shape, X_test.shape

X_train = X_train.reshape(425, 80, 3, 1)
X_test = X_test.reshape(107, 80, 3, 1)


##2D CNN
def CNN():
    model = Sequential()
    model.add(Conv2D(16, (2, 2), activation = "relu", input_shape = X_train[0].shape))
    model.add(Dropout(0.1))
    
    model.add(Conv2D(32, (2, 2), activation = "relu"))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(6, activation = "softmax"))
    
    model.summary()
    
    return model

model = CNN()


model.compile(optimizer=Adam(lr=0.001), loss = "sparse_categorical_crossentropy", metrics = ["accuracy"])
    


history = model.fit(X_train, y_train, epochs = 10, validation_data =(X_test, y_test), verbose = 1)




help(model)

history.history

#Plot training and validation accuracy values
def plot_learningCurve(history, epoch):import pickle
    
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

y_pred = model.predict_classes(X_test)


columns = ["user", "activity", "time", "x", "y", "z"]


#Plot confusion matrix


import matplotlib
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
#to reset matplotli
#matplotlib.rcParams.update(matplotlib.rcParamsDefault)

#matplotlib.rc("font", **font)
mat = confusion_matrix(y_test, y_pred)
fig, ax = plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, 
                                show_normed=True, figsize=(6,6))
                                
plt.tight_layout()
plt.savefig("human_act.png")
plt.show()

model.save_weights("model.h5")

































































































