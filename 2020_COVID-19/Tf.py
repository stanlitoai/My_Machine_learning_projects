import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout,  Flatten, BatchNormalization
from tensorflow.keras.layers import MaxPool2D #, Zeropadding2D
#from tensorflow.keras.proprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.optimizers import SGD


import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt



bas= os.getcwd()
img_dir = os.path.join(bas, "images")
 
train_set_dir = ('images')
test_set_dir = ('dataset/test_set')                                                       
                                                        
                                                     
img_width = 224
img_height = 224
batch_size =5

datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(directory = train_set_dir,
                                              target_size=(img_width, img_height),
#                                              classes= ["dogs", "cats"],
#                                              class_mode= "binary",
                                              batch_size = batch_size,
                                              subset='training')                  
                    

val_generator = datagen.flow_from_directory(directory = train_set_dir,
                                              target_size=(img_width, img_height),
#                                              classes= ["dogs", "cats"],
#                                              class_mode= "binary",
                                              batch_size = batch_size,
                                              subset='validation') 



for image_batch, label_batch in train_generator:
    break


print(train_generato­r.class_indices)


labels = '\n'.join(sorted(tra­in_generator.class_i­ndices.keys()))


model = Sequential()
model.add(Conv2D(filters= 64, kernel_size=(3, 3), activation ="relu", padding= "same",
                 kernel_initializer="he_uniform",
                 input_shape=(img_width, img_height, 3)))

model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(128, activation="relu", kernel_initializer="he_uniform"))
model.add(Dense(13, activation="sigmoid"))


opt = SGD(learning_rate= 0.01, momentum=0.9)

model.compile(optimizer = opt, loss = 'binary_crossentropy', metrics = ['accuracy'])

history = model.fit_generator(generator= train_generator, steps_per_epoch= len(train_generator),
                              epochs = 10, validation_data = val_generator,
                              validation_steps= len(val_generator), verbose = 1)















