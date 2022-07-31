import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential



##CALLBACK

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("loss")<6.4063e-05):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True
            
            
callbacks= myCallback()



X= np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
y= np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


model = Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer="sgd", loss="mean_squared_error")
model.summary()

model.fit(X,y, epochs=10000, callbacks=[callbacks])


print(model.predict([10.0]))

tf.__version__

##ANother lesson...week2

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


 ##ANother lesson...week3

import pandas as pd
import numpy as np
from scipy import misc

i = misc.ascent()
import matplotlib.pyplot as plt

plt.grid(False)
plt.gray()
plt.axis("off")
plt.imshow(i)
plt.show()


i_transferred = np.copy(i)
size_x = i_transferred.shape[0]
size_y = i_transferred.shape[1]


filter = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]





























































