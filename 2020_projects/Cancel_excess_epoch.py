##CALLBACK


import tensorflow as tf

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get("loss")<0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_training = True
            
            
callbacks= myCallback()

model.compile(optimizer = "adam",
              loss = "sparse_categorical_crossentropy",
              metrics = ["accuracy"])

model.fit(train_images, train_labels, epochs=10, callbacks=[callbacks])