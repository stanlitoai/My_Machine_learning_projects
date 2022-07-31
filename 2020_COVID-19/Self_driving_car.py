import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
from sklearn.utils import shuffle
import matplotlib.image as npimg
from imgaug import augmenters as iaa
import cv2
import pandas as pd
import random
import ntpath


datadir = "Data"
columns= ["center", "left", "right", "steering", "throttle", "reverse", "speed"]

data = pd.read_csv("driving_log.csv", names = columns)
pd.set_option("display.max_colwidth", -1)
data.head()


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail

data["center"] = data["center"].apply(path_leaf)
data["left"] = data["left"].apply(path_leaf)
data["right"] = data["right"].apply(path_leaf)
data.head()

num_bins = 25
samples_per_bin = 650
hist, bins = np.histogram(data["steering"], num_bins)
center = (bins[:-1]+bins[1:])*0.5
plt.bar(center, hist, width=0.05)
plt.grid()
plt.plot((np.min(data["steering"]), np.max(data["steering"])), (samples_per_bin, samples_per_bin))
print(hist)

##BALANCING THE DATA 
##seriously, am still confused

print("total data", len(data))
remove_list =[]

for j in range(num_bins):
    list_ = []
    for i in range(len(data["steering"])):
        if data["steering"][i] >= bins[j] and data["steering"][i] <=bins[j+1]:
            list_.append(i)
    list_= shuffle(list_)
    list_=list_[samples_per_bin:]
    remove_list.extend(list_)
    
print("removed", len(remove_list))
data.drop(data.index[remove_list], inplace=True)
print("remaining", len(data))
        
        
hist, _ = np.histogram(data["steering"], num_bins)
plt.bar(center, hist, width=0.05)
plt.grid()
plt.plot((np.min(data["steering"]), np.max(data["steering"])), (samples_per_bin, samples_per_bin))
    
            
  
##TRAINING / VALIDATION    
#Splitting our dataset into the X=image_paths and Y=steerings
print(data.iloc[1])

def load_img_steering(datadir, df):
    image_path = []
    steering = []
    for i in range(len(data)):
        indexed_data = data.iloc[i]
        center, left, right = indexed_data[0], indexed_data[1], indexed_data[2]
        image_path.append(os.path.join(datadir, center.strip()))
        steering.append(float(indexed_data[3]))
    image_paths = np.asarray(image_path)
    steerings = np.asarray(steering)
    return image_paths, steerings

image_paths, steerings = load_img_steering('IMG', data)


##APPLYING TRAIN_TEST_SPLIT

X_train, X_valid, y_train, y_valid = train_test_split(image_paths, steerings, test_size = 0.2, random_state=0)
print("Training Sample: {}\n Valid sample: {}".format(len(X_train), len(X_valid)))


fig, axes = plt.subplots(1, 2, figsize=(10,4))
axes[0].hist(y_train, bins = num_bins, width=.05, color= "blue")
plt.grid()
axes[0].set_title("Training set")

#fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[1].hist(y_valid, bins = num_bins, width=.05, color= "red")
plt.grid()
axes[1].set_title("Validation set")


##defining a zoom function so it can help in zooming the data
#applying some pre preprocessing techniques

def zoom(image):
    zoom  = iaa.Affine(scale = (1, 1.3))
    image = zoom.augment_image(image)
    return image

##TO VISUALIZE THE ZOOMED IMAGE
image = image_paths[random.randint(0, 1000)]
original_image = npimg.imread(image)
zoomed_image = zoom(original_image)

fig, axes = plt.subplots(1, 2, figsize =(10, 8))
fig.tight_layout()


axes[0].imshow(original_image)
axes[0].set_title("Original image")

axes[1].imshow(zoomed_image)
axes[1].set_title("Zoom image")


##image paning is the horizontal or vertivcal paning of image

def pan_image(image):
    pan = iaa.Affine(translate_percent = {'x' : (-0.1, 0.1), 'y' : (-0.1, 0.1)})
    image = pan.augment_image(image)
    return image

##TO VISUALIZE THE PANNED IMAGE
image = image_paths[random.randint(0, 1000)]
original_image = npimg.imread(image)
panned_image = pan_image(original_image)

fig, axes = plt.subplots(1, 2, figsize =(10, 8))
fig.tight_layout()


axes[0].imshow(original_image)
axes[0].set_title("Original image")

axes[1].imshow(panned_image)
axes[1].set_title("Panned image")

##now lets play around the image by making it brighter or darker
    
def brightness(image):
    bright_image = iaa.Multiply((0.2, 1.2))
    image = bright_image.augment_image(image)
    return image

##TO VISUALIZE THE BRIGHTNESS OF THE IMAGE
image = image_paths[random.randint(0, 1000)]
original_image = npimg.imread(image)
brightend_image = brightness(original_image)

fig, axes = plt.subplots(1, 2, figsize =(10, 8))
fig.tight_layout()


axes[0].imshow(original_image)
axes[0].set_title("Original image")

axes[1].imshow(brightend_image)
axes[1].set_title("brightend image")


##NOTE THAT FLIPPING CAN PROVIDE ADDITIONAL BALANCING
def flip_image(image, steering_angle):
    image = cv2.flip(image, 1)
    steering_angle = -steering_angle
    return image, steering_angle


##TO VISUALIZE THE FLIP IMAGE
random_index = random.randint(0, 1000)
image = image_paths[random_index]
steering_angle = steerings[random_index]

original_image = npimg.imread(image)
flipped_image, flipped_steering_angle = flip_image(original_image, steering_angle)

fig, axes = plt.subplots(1, 2, figsize =(10, 8))
fig.tight_layout()

axes[0].imshow(original_image)
axes[0].set_title("Original image - " + "Steering Angle: " + str(steering_angle))

axes[1].imshow(flipped_image)
axes[1].set_title("flipped image - " + "Steering Angle: " + str(flipped_steering_angle))


#
#this code is to ensure that the preprocessing above is done only 50% at a time

def random_augment(image, steering_angle):
    image = npimg.imread(image)
    if np.random.rand() < 0.5:
        image = pan_image(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flip_image(image, steering_angle)
    
    return image, steering_angle

##NOW LETS VISUALIZE IT BY PLOTTING A GRAPH
ncol = 2
nrow = 4

fig, axes = plt.subplots(ncol, nrow, figsize =(15, 50))
fig.tight_layout()

for i in range(10):
    random_ = random.randint(0, len(image_paths) - 1)
    random_image =image_paths[random_]
    random_steering = steerings[random_]
    
    original_image = npimg.imread(random_image)
    augmented_image, steering = random_augment(random_image, random_steering)
    
    axes[i][0].imshow(original_image)
    axes[i][0].set_title("Original image")
    
    axes[i][1].imshow(augmented_image)
    axes[i][1].set_title("Augmented image")
    
%matplotlib

##PREPROCESSING DATA

def img_preprocess(img):
    img = img[60:135, :, :]
    img= cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img= cv2.GaussianBlur(img, (3, 3), 0)
    img = cv2.resize(img, (200, 66))
    img = img/255.0
    return img


image = image_paths[63]
original_image = npimg.imread(image)
preprocessed_image = img_preprocess(original_image)

fig, axes = plt.subplots(1, 2, figsize=(12,8))
axes[0].imshow(original_image)
axes[0].set_title("Original set")

axes[1].imshow(preprocessed_image)
axes[1].set_title("Preprocesssed image")



#Creating a batch generator
def batch_generator(image_paths, steering_ang, batch_size, is_training):
    
    while True:
        batch_img = []
        batch_steering = []
        
        for i in range(batch_size):
            random_index = random.randint(0, len(image_paths) - 1)
            
            if is_training:
                img, steering = random_augment(image_paths[random_index],
                                               steering_ang[random_index])
                                           
            else:
                img = npimg.imread(image_paths[random_index])
                steering = steering_ang[random_index]
                
            img = img_preprocess(img)
            batch_img.append(img)
            batch_steering.append(steering)
        
        yield(np.asarray(batch_img), np.asarray(batch_steering))

##NOW CALLING OUR BATCH GENERATOR USING NEXT FUNCTION
        
X_train_gen, y_train_gen = next(batch_generator(X_train, y_train, 1, 1))
X_valid_gen, y_valid_gen = next(batch_generator(X_valid, y_valid, 1, 0))

##LETS VISUALIZE OUR BATCH GENERATOR

fig, axes = plt.subplots(1, 2, figsize =(10, 8))
fig.tight_layout()


axes[0].imshow(X_train_gen[0])
axes[0].set_title("Training Image")

axes[1].imshow(X_valid_gen[0])
axes[1].set_title("Validation Image")


"""""
##PREPROCESSING OUR TRAINING SET THEN WE'LL MOVE TO VALIDATION

X_train = np.array(list(map(img_preprocess, X_train)))
X_valid = np.array(list(map(img_preprocess, X_valid)))

##visulzing it
plt.imshow(X_train[random.randint(0, len(X_train)-1)])
plt.axis("off")
print(X_train.shape)

"""""


##DEFINING NVIDIA MODEL

def nvidia_model():
    model = Sequential()
    model.add(Conv2D(24, (5, 5), subsample=(2, 2), activation='elu', input_shape=(66, 200, 3)))
    model.add(Conv2D(36, (5, 5), activation='elu', subsample=(2, 2)))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (5, 5), activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='elu'))
    
    model.add(Conv2D(64, (3, 3), activation='elu'))
    model.add(Dropout(0.3))
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
#    model.add(Dropout(0.5))
    
    model.add(Dense(50, activation='elu'))
#    model.add(Dropout(0.5))
    
    
    model.add(Dense(10, activation='elu'))
#    model.add(Dropout(0.5))
    
    
    model.add(Dense(1))
    model.summary()
    
    
    
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=1e-4), metrics = ["accuracy"])

    return model

model = nvidia_model()


##LETS GO TO THE FUN PART NOW BY TRAINING OUR MODEL

history = model.fit_generator(batch_generator(X_train, y_train, 100, 1), 
                              steps_per_epoch = 400,
                              epochs =5,
                              validation_data = batch_generator(X_valid, y_valid, 100, 0),
                               verbose=1, 
                               validation_steps = 300, 
                               shuffle=1)
                    






test_loss, test_acc = model.evaluate(X_valid, y_valid)

print(test_acc)


from sklearn.metrics import accuracy_score

y_pred = model.predict_classes(X_valid)

accuracy_score(y_valid, y_pred)


epoch =30

help(model)

history.history

#Plot training and validation accuracy values


def plot_learningCurve(history, epoch):
    
    #Plot training and validation accuracy values
    epoch_range = range(1, epoch+1)
    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])
    plt.title("Model accuracy")
    plt.ylabel("Accuracy")
    plt.grid()
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()
    
    
    
    #Plot training and validation loss values
    plt.plot( history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("Model loss")
    plt.grid()
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Val"], loc= "upper left")
    plt.show()


plot_learningCurve(history, 30)




#saving the model

model.save('model.h5')


from google.colab import files
files.download('model.h5')


#Visualising the result
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend('training, validation')
plt.grid()
plt.title('loss')
plt.xlabel('Epoch')

































