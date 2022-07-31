import cv2
import os
import glob
import numpy as np

face = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")

#recognizer = cv2.face.LBPHFaceRecognizer_create()

inputFolder = "images"
folderLen = len(inputFolder)
os.mkdir("Resized1") 

for img in glob.glob(inputFolder + "/*.jpg"):
    image = cv2.imread(img)
    imgResized= cv2.resize(image, (150,150))
    cv2.imwrite("Resized1" + img[folderLen:], imgResized)
    cv2.imshow("image", imgResized)
    cv2.waitKey(30)

cv2.destroyAllWindows()

img = cv2.imread("my_face.png")

while (True):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #print(gray)

    #left_eye = left_eye.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=5)
    faces = face.detectMultiScale(img, scaleFactor= 1.2, minNeighbors=5)
    #right_eye = right_eye.detectMultiScale(gray, scaleFactor= 1.5, minNeighbors=5)
    for (x,y,w,h) in faces:
#        print(x,y,w,h)
        roi_gray = array[y:y+h, x:x+w]
#        roi_color = frame[y:y + h, x:x + w]
        
        color= (255,0,0)
        stroke= 2
        end_cord_x= x + w
        end_cord_y= y + h
        a=cv2.rectangle(array,(x,y), (end_cord_x, end_cord_y), color,stroke)
#        img_item = "my_face.png"
       # cv2.imwrite(img_item, a)
        
        #Display the result
    cv2.imshow("frame", roi_gray)
    if cv2.waitKey(20) & 0xFF == ord("q"):
        break

#cap.release()
cv2.destroyAllWindows()





for file in os.listdir("people"):
    print(file)
    full_path = "people/" + file
    print(full_path)
    for i, person in enumerate(people):
        
        labels_dic[i] = person
        for image in os.listdir("people/­" + person):
#            images.append(cv2.im­read("people/" + person + '/' + image, 0))
            labels.append(person­)
    
        return (images, np.array(labels), labels_dic)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
name = "stanley"
passwrd = 123434
count = 3

while count > 0:
    username= input("Enter your username: ")
    password= input("Enter your password: ")
    
    if username == name:
        print("welcome "+ username)
        break
        
    else:
        print("incorrect password ")
        count-=1
        
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    