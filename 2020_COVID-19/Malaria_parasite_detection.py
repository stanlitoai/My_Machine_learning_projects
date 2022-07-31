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

from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
print(tf.__version__)

img_width = 100
img_height = 100


datagen = ImageDataGenerator(rescale = 1./255, validation_split=0.2)

train_data_gen = datagen.flow_from_directory(directory = "training",
                                             target_size =  (img_width, img_height),
                                             class_mode ="binary",
                                             batch_size = 16,
                                             subset = "training")

train_data_gen.labels



valid_data_gen = datagen.flow_from_directory(directory = "training",
                                             target_size =  (img_width, img_height),
                                             class_mode ="binary",
                                             batch_size = 16,
                                             subset = "validation")

valid_data_gen.labels


##CNN

def cnn_model():
    model = Sequential()
    model.add(Conv2D(16, (3, 3), input_shape = (img_width, img_height, 3),
                     activation= "relu"))
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.3))
    
    
    model.add(Conv2D(32, (3, 3), activation= "relu"))                   
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))
    
    
    model.add(Conv2D(64, (3, 3), activation= "relu"))                   
    model.add(MaxPool2D(2, 2))
    model.add(Dropout(0.5))

    
    
    model.add(Flatten())
    model.add(Dense(64, activation = "relu"))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation = "sigmoid"))
    
    model.summary()
    
    return model


model = cnn_model()


model.compile(optimizer = Adam(lr=0.0005), loss = "binary_crossentropy",
                metrics = ["accuracy"])


history = model.fit_generator(generator = train_data_gen,
                              steps_per_epoch = len(train_data_gen),
                              epochs = 10,
                              validation_data = valid_data_gen,
                              validation_steps = len(valid_data_gen))
                                                    
                                                    
                                                    
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





model.save("reco.h5")




from keras.preprocessing import image
test_image = image.load_img('images/pred/chi.jpg', target_size = (img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train_data_gen.class_indices
if result[0][0] == 1:
    prediction = 'stan'
else:
    prediction = 'cat'
















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





from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Activation

from sklearn.model_selection import train_test_split
from sklearn import preprocessing


K.set_image_data_format('channels_last')
np.random.seed(0)








def create_model(input_shape, with_summary):
    model = Sequential()
    model.add(Conv2D(10, kernel_size=5, padding="same", input_shape=input_shape, activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(20, kernel_size=3, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(30, kernel_size=3, padding="same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    #model.add(Conv2D(500, kernel_size=3, padding="same", activation = 'relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    #model.add(Conv2D(1024, kernel_size=3, padding="valid", activation = 'relu'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(units=30, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(units=5, activation='relu'))
    #model.add(Dropout(0.1))

    model.add(Dense(13))
    model.add(Activation("sigmoid"))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    if with_summary:
        model.summary()

    return model

def save_history(hist, filepath):
    with open(filepath, 'w') as f:
        json.dump(hist.history, f)

def plot_loss(history_filepath):
    with open(history_filepath) as json_data:
        history = json.load(json_data)
        #print(history)
    print(history.keys())
    plt.plot(history['loss'])
    plt.plot(history['acc'])
    plt.title('Training metrics')
    #plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['loss', 'accuracy'], loc='upper left')
    plt.show()


input_shape = (img_width, img_height, 3)
model = create_model(input_shape=input_shape, with_summary=True)


hist = model.fit(X_train, y_train,batch_size=512,epochs=20)



history = model.fit_generator(generator = train_data_gen,
                              steps_per_epoch = len(train_data_gen),
                              epochs = 20,
                              validation_data = valid_data_gen,
                              validation_steps = len(valid_data_gen))


















