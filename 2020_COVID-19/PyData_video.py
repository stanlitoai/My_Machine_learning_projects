import numpy as np
import os
import math
import matplotlib.pyplot as plt
#%matplotlib
import cv2

webcam = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(*"XVID")
face = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")
#video = cv2.VideoWriter("video/kelly.avi", fourcc, 20.0, (640,480))
#cv2.namedWindow("stanlito tutorial", cv2.WINDOW_NORMAL)
#cv2.resizeWindow("stanlito tutorial", 850,450)


while(True):
    ret, frame = webcam.read()
    #video.write(frame)
    #plt.show(frame)
#    if not ret:
#        break

    #container = cv2.VideoWriter(wiz, fourcc, frames_per_second(float), pixel_size(tuple))
    #container.write(frame)
    #flag= cv2.CASCADE_FIND_BIGGEST_OBJECT
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_ = cv2.equalizeHist(gray)

    faces = face.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5, minSize= (30, 30))
    for (x,y,w,h) in faces:
        print(x,y,w,h)
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y + h, x:x + w]
        cv2.rectangle(frame, (x, y), (x+w,y+h), (255,0,0), 5)

        img_item = "my_fac.png"
        cv2.imwrite(img_item, roi_gray)
        cv2.imshow("stanlito tutorial", frame)

    #plt.imshow(gray)
    #plt.imshow(gray_)
    plt.show()


    #cv2.startWindowThread()
    #cv2.namedWindow("stanlito", cv2.WINDOW_NORMAL)
 #
#    frame = cv2.flip(frame, 1)






    if cv2.waitKey(20) & 0xFF == ord("q"):

        break

webcam.release()
cv2.destroyAllWindows()