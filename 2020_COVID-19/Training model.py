import os
import cv2
import numpy as np
import imutils
import argparse
import pickle


def collect_dataset():
    img = []
    labels = []
    labels_dic = {}
    people = [people for person  in os.listdir("images") ]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("images" + person):
            img.append(cv2.imread("images" + person + "/" +  image, 3))
            labels.append(i)


    return  (img, np.array(labels), labels_dic)

    print(type(img))


rec_ =