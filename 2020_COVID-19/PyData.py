import numpy as np
import os
import math
import matplotlib.pyplot as plt
#%matplotlib
import cv2

cap = cv2.VideoCapture(0)
#fourcc = cv2.VideoWriter_fourcc(video_codec)

while(True):
    ret, frame = cap.read()
    print (frame)

    #container = cv2.VideoWriter(wiz, fourcc, frames_per_second(float), pixel_size(tuple))
    #container.write(frame)


    #cv2.startWindowThread()
    #cv2.namedWindow("stanlito", cv2.WINDOW_NORMAL)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    img_item = "my_facy.png"
    cv2.imwrite(img_item, gray)
    cv2.imshow("frame", frame)




    if cv2.waitKey(20) & 0xFF == ord("q"):

        break

cap.release()
cv2.destroyAllWindows()

#plt.imshow(gray)
#plt.axis("off")
#plt.show()


#def plt_show(image, title=""):
#    if len(image.shape) == 3:
#        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#    plt.axis("off")
#    plt.title(title)
#    plt.imshow(image, cmap=None)
#    plt.show()