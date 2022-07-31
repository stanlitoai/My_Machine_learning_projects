from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from tensorflow.keras.preprocessing import image
import os
import numpy as np


"""""
###################################################
cat =os.path.join('training_set/cats')
cats = os.listdir(cat)

dog =os.path.join('training_set/dogs')
dogs = os.listdir(dog)
oya =[]
for i in  dogs:
    oya.append(i[:5])
    
 
    plt.imshow(oya[1])
    plt.show()
print(cats[:10])
print(dogs[:10])

##Graph
nrows =4
ncols=4

pix_index=0

fig = plt.gcf()
fig.set_size_inches(ncols *4, nrows *4)
pix_index +=8

next_cat_pix = [os.path.join('training_set/cats', fname)
                for fname in cats[pix_index -8: pix_index]]

next_dog_pix = [os.path.join('training_set/dogs', fname)
                for fname in dogs[pix_index -8: pix_index]]


for i, img_path in enumerate[next_cat_pix+next_dog_pix]:
    sp = plt.subplot(nrows, ncols, i + 1)
    sp.axis("off")
    
    im = npimg.imread(img_path)
    plt.imshow(im)
    
plt.show()
###############################################################

"""""



def Image_gen():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation="relu", input_shape = (350, 350, 3)))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(32, (3,3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(2, 2))
    model.add(Flatten())
    model.add(Dense(512, activation = "relu"))
    model.add(Dense(12, activation = "softmax"))
    
    model.summary()
    
    return model
    
    
model = Image_gen()
"""""
datagen = ImageDataGenerator(featurewise_center=True,
                            featurewise_std_normalization=True,
                             rotation_range=20,
                             width_shift_range=0.2,
                             height_shift_range=0.2,
                             horizontal_flip=True)


"""""




from tensorflow.keras.optimizers import RMSprop

model.compile(loss= "sparse_categorical_crossentropy", optimizer = RMSprop(lr=0.001), metrics=["acc"])


train_generator = ImageDataGenerator(rescale = 1./255,
                                     shear_range=0.2,
                                     zoom_range=0.2,
                                     horizontal_flip=True,
                                     vertical_flip=True,
                                     validation_split=0.2)


train_generator = train_generator.flow_from_directory(directory = "pics",
                                                    target_size=(350, 350),
                                                    batch_size=32,
                                                    class_mode="binary",
                                                    subset = "training")


test_datagen = ImageDataGenerator(rescale=1./255,
                                  validation_split=0.2)

test_generator = test_datagen.flow_from_directory(directory = "pics",
                                                    target_size=(350, 350),
                                                    batch_size=32,
                                                    class_mode="binary",
                                                    shuffle=False,
                                                    subset = "validation")






history = model.fit_generator(train_generator,
                              steps_per_epoch=len(train_generator),
                              epochs=40,
                              validation_data =test_generator,
                              validation_steps=len(test_generator))

model.save("cats_and_dogs_2020.h5")




##PREDICTION
###
from tqdm import tqdm
X= []
for i in tqdm(range(data.shape[0])):
    path = "pred\\"+ data["image"][i]
    img = image.load_img(path, target_size = (350, 350, 3))
    img =image.img_to_array(img)
    img = img/255.0
    X.append(img)


test_image = image.load_img("pred/noll.jpg",
                            target_size = (350, 350))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result[0])
train_generator.class_indices
if result[0]==0:
    print(cn + ": is jammy")
    
else:
    print(cn + ": is merlin")

epochs=40

def plot_learningCurve(history, epoch):
    
    #Plot training and validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(epoch_range, history.history["acc"])
    plt.plot(epoch_range, history.history["val_acc"])
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


plot_learningCurve(history, epochs)





















