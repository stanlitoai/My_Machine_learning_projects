import os
import numpy as np
import cv2
from  PIL import Image
import pickle
import matplotlib.pyplot as plt




Base_dir = os.path.dirname(os.path.abspath(__file__))
img_dir = os.path.join(Base_dir, "images")

face = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
eye = cv2.CascadeClassifier("cascades/data/haarcascade_eye.xml")



recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
x_train = []
y_labels = []
for root, dirs, files in os.walk(img_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path = os.path.join(root,file)
            label = os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
            print(label, path)
            if not label in label_ids:
                label_ids[label] = current_id
                current_id +=1
            id_ = label_ids[label]
            print(label_ids)


            pil_image = Image.open(path).convert("L") #grayscale
            size = (280, 280)
            final_image = pil_image.resize(size, Image.ANTIALIAS)

            image_array = np.array(final_image, "uint8")
            print(image_array)
            faces = face.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)


            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
#                eyes = eye.detectMultiScale(roi)
#                for (ex, ey, ew, eh) in eyes:
                    # print(x,y,w,h)
#                    cv2.rectangle(roi, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)
                    x_train.append(roi)
                    y_labels.append(id_)




            #x_train.append(path)
            #y_labels.append(label)


#print(y_labels)
#print(x_train)





with open("labels.pickle", 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, np.array(y_labels))
recognizer.save("face-trainner.yml")
print("Training set is complete now!!")
