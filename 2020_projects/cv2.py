import cv2.cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
 

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, 100)

while True:
    _, frame = cap.read()
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    
cap.release()
cv2.destroyAllWindows()



#############chapter2##########

img = cv2.imread("g.jpg")
kernal = np.ones((5,5), np.uint8)

imggray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (7, 7), 0)


##Edge detectot
imgcanny = cv2.Canny(img, 100, 100)
imgdialate = cv2.dilate(imgcanny, kernal, iterations=1)
imgerode = cv2.erode(imgdialate, kernal, iterations=1)

cv2.imshow("Blur", imgblur)
cv2.imshow("Gray", imggray)
cv2.imshow("Canny", imgcanny)
cv2.imshow("dialate", imgdialate)
cv2.imshow("erode", imgerode)
cv2.waitKey(0)



############chapeter3######RESIZE and crop images

img = cv2.imread("g.jpg")
img.shape
imgresize = cv2.resize(img, (480, 680))

imgcrop = img[0:400, 250: 550 ]
#plt.imshow(img)
#plt.show()
cv2.imshow("img", img)
cv2.imshow("img resize", imgresize)
cv2.imshow("img crop", imgcrop)
cv2.waitKey(0)


############chapter4#####Draw line and text on images

img = np.zeros((512, 512, 3), np.uint8)
img = cv2.imread("chi.png")
#img[:]= 255, 0, 0
imgresize = cv2.resize(img, (400, 580))

imgcrop = img[200:1200, : ]
#cv2.line(img, (0,0), (img.shape[1], img.shape[0]), (0, 255, 0), 3)
rec=cv2.rectangle(img, (250,700), (450, 400), (0, 0, 255), 3)
#cv2.circle(img, (300, 300), 200, (255, 255,0), 5)
cv2.putText(img, "MR PRESIDENT", (250,700), cv2.FONT_HERSHEY_COMPLEX, .9, (0, 150,0), 2)
#plt.imshow(imgcrop)
#plt.show()

cv2.imshow("img", imgcrop)
cv2.imwrite("stan.png", imgcrop)
cv2.waitKey(0)


###############chapter5###########WARP PRESPECTIVE

img = cv2.imread("cards.png")


cv2.imshow("img",img)

cv2.waitKey(0)


###############chapter6###########JOIN IMAGES

img = cv2.imread("cards.png")

imghor = np.hstack((img, img))
imgver = np.vstack((img, img))

cv2.imshow("imghor",imghor)
cv2.imshow("imgver",imgver)

cv2.waitKey(0)


###############chapter7###########COLOR DETECTION


def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars", 640, 240)
cv2.createTrackbar("Hue Min", "TrackBars", 0, 179, empty)
cv2.createTrackbar("Hue Max", "TrackBars", 47, 179, empty)
cv2.createTrackbar("Sat Min", "TrackBars", 69, 255, empty)
cv2.createTrackbar("Sat Max", "TrackBars", 171, 255, empty)
cv2.createTrackbar("Val Min", "TrackBars", 0, 255, empty)
cv2.createTrackbar("Val Max", "TrackBars", 255, 255, empty)


while True:
    
    img = cv2.imread("g.jpg")
    imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    h_min = cv2.getTrackbarPos("Hue Min", "TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min, h_max, s_min, s_max, v_min, v_max)
    lower = np.array([h_min, s_min, v_min])
    upper = np.array([h_max, s_max, v_max])
    
    mask = cv2.inRange(imghsv, lower, upper)
    imgresult = cv2.bitwise_and(img, img, mask=mask)
    
    cv2.imshow("img",img)
    cv2.imshow("imghsv",imghsv)
    cv2.imshow("masked",mask)
    cv2.imshow("imgresult", imgresult)
    
    cv2.waitKey(1)
cv2.destroyAllWindows()




###############chapter8###########CONTOURS/ SHAPE DETECTION



def stackImages(scale, imgArray):  
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
           
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor( imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver


def getcontours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 400:
            cv2.drawContours(imgcontour, cnt, -1, (255, 0, 0), 3)
            peri = cv2.arcLength(cnt, True)
            #print(peri)
            approx = cv2.approxPolyDP(cnt, 0.02*peri, True)
            print(len(approx))
            objcor = len(approx)
            x, y, w, h = cv2.boundingRect(approx)
            
            if objcor == 3:
                objectType="Tri"
                
            elif objcor == 4:
                aspRatio = w/float(h)
                if aspRatio > 0.95 and aspRatio<1.05:
                    objectType="Square"
                else:
                    objectType="Triangle"
          
            elif objcor == 6:
                objectType ="Pentagon"
                
            elif objcor == 10:
                objectType ="Stars"
                
            elif objcor == 8:
                objectType ="Circle"
    
            else:
                objectType="None"
            
            cv2.rectangle(imgcontour, (x,y),(x+w, y+h), (0,255,0),2)
            cv2.putText(imgcontour, objectType,
                        (x+(w//2)-10, y+(h//2)-10), cv2.FONT_HERSHEY_COMPLEX,0.7,
                         (0,0,0),2)
                

img = cv2.imread("shapes.png")
imgcontour = img.copy()
imgblank = np.zeros_like(img)
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray, (7,7), 1)
imgcanny = cv2.Canny(imgblur,50, 50)
getcontours(imgcanny)
imgstack = stackImages(0.6, ([imggray, imgblur], [imgcanny, imgcontour]))


#cv2.imshow("img",img)
#cv2.imshow("imggray",imggray)
cv2.imshow("imgstack",imgstack)
#cv2.imshow("canny",imgcanny)


cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()




#img = cv2.imread('chi.png')
#imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)





###############chapter9###########face DETECTION


import cv2.cv2 as cv2


facecascade = cv2.CascadeClassifier("data/haarcascade_frontalface_default.xml")
img = cv2.imread("g.jpg")
imggray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = facecascade.detectMultiScale(imggray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)

cv2.imshow("frame", img)
cv2.waitKey(0)




###############Project###########VIRTUAL PAINT























































































































































































