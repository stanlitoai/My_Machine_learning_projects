from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os

model= VGG16()

model.summary()

for file in os.listdir("img"):
    print(file)
    full_path = "sample/" + file
#    print(full_path)
    
    image = load_img(full_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image =  preprocess_input(image)
    y_pred = model.predict(image)
    label = decode_predictions(y_pred, top =1)
    print(label)
    print()

















