import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import  tensorflow as tf
import zipfile
import keras
from collections import  defaultdict
from  io import StringIO
import matplotlib.pyplot as plt
from  PIL import Image
import DLLs
import cv2
from tensorflow import keras

cap =  cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    image_no_expanded = np.expand_dims(frame, axis= 0)


    cv2.imshow("object detection", cv2.resize(frame,(800,600)))

    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        break

from gym import wrappers

#sys.path.append()



import gym
import py


#print("the version is ",tf.__version__)