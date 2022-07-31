import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.optimizers import Adam
import keras.backend as k
import matplotlib.pyplot as plt
import numpy as np
import random



img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 20
dropout = 0.5


(X_train, y_train), (X_test, y_test) = mnist.load_data()

orig_test = X_test



if k.image_data_format() == "chinnels_first":
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)


X_train = X_train / 255.0
X_test = X_test / 255.0

print(X_train.shape[0], "Training samples")
print(X_test.shape[0], "Test samples")



sizelist = [100, 200, 500, 1000, 2000, 5000, 10000]
accuracy = []

for trainingsize in sizelist:
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = input_shape))
    
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout/2))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(dropout))
    
    
    model.add(Dense(num_classes, activation = "softmax"))
    
    
    model.compile(loss= "sparse_categorical_crossentropy", optimizer = Adam(), metrics =["accuracy"])
    X = X_train[:trainingsize]
    y = y_train[:trainingsize]
    
    history = model.fit(X, y, batch_size = batch_size, epochs = epochs, verbose = 0)
                  
    
    score = model.evaluate(X_test, y_test, verbose =0)
    print("Training sample: %d, test accuracy: %.2f %%" % (trainingsize, score[1]*100))
    accuracy.append(score[1])
    
    
    
    
plt.plot(sizelist, accuracy)
plt.title("Model accuracy")
plt.ylabel("Test Accuracy")
plt.grid()
plt.xlabel("Training Samples")
plt.show()





##LETS TRY THIS METHOD


sizelist = [10, 20, 50, 100, 200, 500, 1000, 2000]
accuracy = []

#choose classes
classes = [8, 9]

#Training set
xtr = []
ytr = []

for i in range(len(y_train)):
    if y_train[i] in classes:
        xtr.append(X_train[i])
        ytr.append(classes.index(y_train[i]))



xtr = np.asarray(xtr)
ytr = np.asarray(ytr)



#Test set
xte = []
yte = []

for i in range(len(y_test)):
    if y_test[i] in classes:
        xte.append(X_test[i])
        yte.append(classes.index(y_test[i]))



xte = np.asarray(xte)
yte = np.asarray(yte)





for trainingsize in sizelist:
    model = Sequential()
    model.add(Conv2D(32, kernel_size = (3, 3), activation = "relu", input_shape = input_shape))
    
    model.add(Conv2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(dropout/2))
    model.add(Flatten())
    model.add(Dense(128, activation = "relu"))
    model.add(Dropout(dropout))
    
    
    model.add(Dense(num_classes, activation = "softmax"))
    
    
    model.compile(loss= "sparse_categorical_crossentropy", optimizer = Adam(), metrics =["accuracy"])
    X = xtr[:trainingsize]
    y = ytr[:trainingsize]
    
    history = model.fit(X, y, batch_size = batch_size, epochs = epochs, verbose = 0)
                  
    
    score = model.evaluate(xte, yte, verbose =0)
    print("Training sample: %d, test accuracy: %.2f %%" % (trainingsize, score[1]*100))
    accuracy.append(score[1])




plt.plot(sizelist, accuracy)
plt.title("Model accuracy")
plt.ylabel("Test Accuracy")
plt.grid()
plt.xlabel("Training Samples")
plt.show()






























