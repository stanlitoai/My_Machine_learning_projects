import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import random



img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 10
epochs = 20
dropout = 0.5


(X_train, y_train), (X_test, y_test) = mnist.load_data()


orig_test = X_test.copy()


X_train = X_train / 255.0
X_test = X_test / 255.0


index = random.randrange(1000)

plt.imshow(orig_test[index], cmap="gray")
plt.title("Label: %d" % y_test[index])
plt.show()



def mnist():
    model = Sequential()
    model.add(Flatten(input_shape = (img_rows, img_cols)))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(dropout))
    model.add(Dense(512, activation = "relu"))
    model.add(Dropout(dropout))
    
    model.add(Dense(num_classes, activation = "softmax"))
    
    model.summary()
    
    return model
    
    
model = mnist()


model.compile(loss= "sparse_categorical_crossentropy", optimizer = Adam(), metrics =["accuracy"])

history = model.fit(X_train, y_train, batch_size = batch_size, epochs = 10, verbose = 1, 
                    validation_data =(X_test, y_test))


#Evaluate model against test data

score = model.evaluate(X_test, y_test, verbose =0)
print("Test loss: ", score[0])
print("Test accuracy: ", score[1])




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



##Svae model structure and trained weights biasses to separate files

model_structure = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_structure)
    
model.save_weights("weights.h5")    




##MY BEST PART OOOOOO
##PREDICTION

from keras.models import model_from_json

with open("model.json", "r") as file:
    loaded_model_json = file.read()
    
mode = model_from_json(loaded_model_json)

mode.load_weights("weights.h5")




(X_train, y_train), (X_test, y_test) = mnist.load_data()


orig_test = X_test.copy()


X_train = X_train / 255.0
X_test = X_test / 255.0


pred = mode.predict(X_test)
most_likely = pred.argmax(1)



index = random.randrange(1000)

plt.imshow(orig_test[index], cmap="gray")
plt.title("prediction: %d, Label: %d" % (most_likely[index], y_test[index]))
plt.show()



##Error Analysis

for i in range(10000):
    index = random.randrange(10000)
    if most_likely[index] !=  y_test[index]:
        break
    
    

plt.imshow(orig_test[index], cmap="gray")
plt.title("prediction: %d, Label: %d" % (most_likely[index], y_test[index]))
plt.show()


plt.bar(range(10), pred[index], tick_label=range(10))
plt.title("Prediction values")
plt.show()




##The following cell calculates the error rate by comparing predicted values.
##This is what keras does when evaluating a model

total = 0.0
misclassified = 0.0 

for  i in range(10000):
    total += 1
    if most_likely[i] != y_test[i]:
        misclassified += 1
        
        
print("Error rate: %.2f %%" % (100.0*misclassified/total))





















