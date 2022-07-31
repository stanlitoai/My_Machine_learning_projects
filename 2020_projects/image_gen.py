from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from tensorflow.keras.preprocessing import image
import os
import numpy as np


train_dr='training_set'
validation_dr='test_set'



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





def Image_gen():
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation="relu", input_shape = (200, 200, 3)))
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
    model.add(Dense(1, activation = "sigmoid"))
    
    model.summary()
    
    return model
    
    
model = Image_gen()


from tensorflow.keras.optimizers import RMSprop

model.compile(loss="binary_crossentropy", optimizer = RMSprop(lr=0.001), metrics=["acc"])



train_datagen = ImageDataGenerator(rescale = 1./255,
                                   horizontal_flip = True)

train_generator = train_datagen.flow_from_directory(train_dr,
                                                    target_size=(200, 200),
                                                    batch_size=128,
                                                    class_mode="binary")


test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(validation_dr,
                                                    target_size=(200, 200),
                                                    batch_size=128,
                                                    class_mode="binary")






history = model.fit_generator(train_generator,
                              steps_per_epoch=10,
                              epochs=25,
                              verbose=2,
                              validation_data =test_generator,
                              validation_steps=10)






##PREDICTION
import numpy as np
from tensorflow.keras.preprocessing import image
import cv2.cv2 as cv2
###
img = 'single_prediction/cat_or_dog_3.jpg'
for fn in img.keys():
    print(fn)
    

    
img=[]
classname = []

paths =os.path.join('single_prediction')
mylist = os.listdir(paths)
#print(mylist)
for cl in mylist:
    classname.append(os.path.splitext(cl)[0])
    print(cl)
    img.append(cl)
    
test_image = image.load_img(cl,
                            target_size = (200, 200))

test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

images = np.vstack([test_image])

classes = model.predict(images, batch_size=10)

print(classes[0])
##training_set.class_indices
for cn in classname:
    print()
if classes[0][0]>0.5:
    print(cn + "is a cat")
    
else:
    print(cn + "is a dog")





test_image = image.load_img('single_prediction/cat_or_dog_3.jpg',
                            target_size = (200, 200))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result[0])
train_generator.class_indices
if result[0]==1:
    print(cn + ": is a dog")
    
else:
    print(cn + ": is a cat")





























