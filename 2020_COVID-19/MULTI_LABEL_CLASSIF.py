import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

from sklearn.model_selection import train_test_split
from tqdm import tqdm

pa = "blood_cell"
data = pd.read_csv("annotations.csv")

img_weight = 350
img_height = 350

X= []
for i in tqdm(range(data.shape[0])):
    path = "images\\"+ data["image"][i]
    img = image.load_img(path, target_size = (img_weight, img_height, 3))
    img =image.img_to_array(img)
    img = img/255.0
    X.append(img)
    
    
X = np.array(X)

X.shape

plt.imshow(X[2])

data["label"][2]

y = data.drop(["image", "label"], axis=1)
y = y.to_numpy()
y.shape


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= .2, random_state = 0)


##BUILD CNN

#def cnn():
model = Sequential()
model.add(Conv2D(16, (3, 3), activation = "relu", input_shape = X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.2))


model.add(Conv2D(32, (3, 3), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.4))

model.add(Conv2D(128, (3, 3), activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPool2D(2, 2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(128, activation= "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation= "relu"))
model.add(BatchNormalization())
model.add(Dropout(0.5))


model.add(Dense(4, activation = "sigmoid"))

model.summary()
    
model.compile(optimizer="adam", loss = "binary_crossentropy", metrics = ["accuracy"])
    
    
#model =cnn()

history = model.fit(X_train, y_train, epochs = 5, validation_data =(X_test, y_test))






##Testing of our model

img = image.load_img(path, target_size = (img_weight, img_height, 3))
plt.imshow(img)
img =image.img_to_array(img)
img = img/255.0
X.append(img)


img = img.reshape(1, img_width, img_height, 3)

classes =data.columns[2:]

y_prob = model.predict(img)

top3 = np.argsort(y_prob[0])[:-4: -1]

for i in range(3):
    print(classes[top3[i]])

































































