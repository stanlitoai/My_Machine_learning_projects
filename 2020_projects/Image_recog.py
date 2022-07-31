import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
print(tf.__version__)

img_width = 100
img_height = 100


datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)

train_data_gen = datagen.flow_from_directory(directory = "Train",
                                             target_size =  (img_width, img_height),
                                             class_mode ="binary",
                                             batch_size = 16,
                                             subset = "training")

train_data_gen.labels



valid_data_gen = datagen.flow_from_directory(directory = "Test",
                                             target_size =  (img_width, img_height),
                                             class_mode ="binary",
                                             batch_size = 16,
                                             subset = "validation")

valid_data_gen.labels

dropout = 0.2


model = Sequential()
model.add(Conv2D(32,(3, 3), activation = "relu", input_shape =(img_width, img_height, 3)))


model.add(Conv2D(64, (3, 3), activation = "relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(dropout))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(Dense(128, activation = "relu"))
model.add(Dropout(0.5))


model.add(Dense(12, activation = "softmax"))
model.summary()

model.compile(loss= "binary_crossentropy", optimizer = Adam(lr = 0.0004), metrics =["accuracy"])


history = model.fit_generator(generator = train_data_gen,
                              steps_per_epoch = len(train_data_gen),
                              epochs = 30,
                              validation_data = valid_data_gen,
                              validation_steps = len(valid_data_gen))





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


plot_learningCurve(history, 5)





from keras.preprocessing import image
test_image = image.load_img('images/pred/stan.jpg', target_size = (img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
a=train_data_gen.class_indices
print(result, a)



















