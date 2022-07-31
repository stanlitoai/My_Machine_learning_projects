import numpy as np
import pandas as pd
import cv2.cv2 as cv2
import os
import matplotlib.pyplot as plt



facecascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
path = "chidera"
images = []
classname = []

stan = pd.DataFrame({"image":images})


mylist = os.listdir(path)
print(mylist)
for cl in mylist:
    img = cv2.imread(f"{path}/{cl}")
    images.append(img)
    classname.append(os.path.splitext(cl)[0])
next_=0
if next_==0:
    next_+=1
    print(next_)
    if next_==10:
        break
    
    for i in range(len(images)):
        plt.imshow(images[next_+=1])
        plt.show()



print(images)
print(classname)

images[1] = cv2.resize(images[1], (500, 680))
cv2.imshow("stan", images[1])
cv2.waitKey(0)

def recog(images):
    findfaces = []
    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        faces = facecascade.detectMultiScale(im, 1.1, 4)
        findfaces.append(faces)
    return findfaces











































