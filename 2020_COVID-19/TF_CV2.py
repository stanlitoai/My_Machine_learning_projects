 import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt



(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()



train_images = train_images/255.0
test_images = test_images / 255.0


class_names = ["Airplane", "Automoblie", "Bird", "Cat", "Deer", "Dog",
               "Frog", "Horse", "Ship", "Truck"]


IMG_INDEX = 1

plt.imshow(train_images[1], cmap = plt.cm.binary)
plt.xlabel(class_names[train_labels[1][0]])
plt.show()


model = keras.Sequential()
model.add(layers.Conv2D(32, (3,3), activation = "relu", input_shape =(32, 32, 3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation = "relu"))

model.summary()


model.add(layers.Flatten())
model.add(layers.Dense(64, activation = "relu"))
model.add(layers.Dense(10))


model.compile(optimizer = "adam",
              loss= tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True)
              #loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

history = model.fit(train_images, train_labels, epochs=4,
                    validation_data = (test_images, test_labels))




test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)

print("Test accuracy: ", test_acc)




















