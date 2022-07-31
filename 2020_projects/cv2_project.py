import numpy as np
import cv2.cv2 as cv2
#import face_recognition
from datetime import datetime



facecascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
train_img = cv2.imread("chi.jpg")
train_img = cv2.resize(train_img, (500, 680))
train_img = cv2.cvtColor(train_img, cv2.COLOR_BGR2RGB)


test_img = cv2.imread("wizzy.jpg")
test_img = cv2.resize(test_img, (500, 680))
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

##detect face
faces = facecascade.detectMultiScale(train_img, 1.1, 4)
faces1 = facecascade.detectMultiScale(test_img, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(train_img, (x,y),(x+w, y+h), (255,0,255), 2)

for (x,y,w,h) in faces1:
    cv2.rectangle(test_img, (x,y),(x+w, y+h), (255,0,255), 2)
  


cv2.imshow("train", train_img)
cv2.imshow("test", test_img)
cv2.waitKey(0)




##ATTENDANCE

import numpy as np
import cv2.cv2 as cv2
import face_recognition
import os


path = "chidera"
images = []
classname = []

mylist = os.listdir(path)
#print(mylist)
for cl in mylist:
    img = cv2.imread(f"{path}/{cl}")
    images.append(img)
    classname.append(os.path.splitext(cl)[0])

print(classname)


def recog(images):
    findfaces = []
    for im in images:
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        faces = facecascade.detectMultiScale(im, 1.1, 4)
        findfaces.append(faces)
    return findfaces
 

def markattendance(name):
    with open("attendances.csv", "r+") as f:
        myDatalist = f.readlines()
        nameList = []
        for line in myDatalist:
            entry = line.split()
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime("%H:%M:%S")
            f.writelines(f"\n{name}, {dtString}")


       
        
knownfaces = recog(images)

print(len(knownfaces))


cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    imgs = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    faces = facecascade.detectMultiScale(imgs, 1.1, 4)
    
    
    for people in zip(faces):
        matches = []
        #predicting the faces found in the webcam
        
        #finding the highest predicted value in  the webcam
        matchindex = np.argmax(matches)
        
        if pred>90:
            name = classaname[matchindex].upper()
            print(name)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y),(x+w, y+h), (255,0,255), 2)
                markattendance(name)
            
            
            
    cv2.imshow("webcam", frame)
    cv2.waitKey(1)
    
    




















































