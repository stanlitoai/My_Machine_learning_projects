import cv2
import numpy as np
import os

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

images, labels, labels_dic = dataset()




def dataset():
    images = []
    labels = []
    labels_dic = {}
    people = [person for person in os.listdir("people/­")]
    for i, person in enumerate(people):
        labels_dic[i] = person
        for image in os.listdir("people/­" + person):
            images.append(cv2.im­read("people/" + person + '/' + image, 0))
            labels.append(person­)
    
        return (images, np.array(labels), labels_dic)

images, labels, labels_dic = dataset()



class FaceDetector(object)­:
    def __init__(self, xml_path):
        self.classifier = cv2.CascadeClassifie­r(xml_path)
    
    def detect(self, image, biggest_only=True):
        scale_factor = 1.2
        min_neighbors = 5
        min_size = (30, 30)
        biggest_only = True
        faces_coord = self.classifier.dete­ctMultiScale(image,
        scaleFactor=scale_fa­ctor,
        minNeighbors=min_nei­ghbors,
        minSize=min_size,
        flags=cv2.CASCADE_SC­ALE_IMAGE)
        return faces_coord

def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:
        w_rm = int(0.3 * w / 2)
        faces.append(image[y­: y + h, x + w_rm: x + w - w_rm])
        
    return faces

def resize(images, size=(224, 224)):
    images_norm = []
    for image in images:
        if image.shape < size:
            image_norm = cv2.resize(image, size,
            interpolation=cv2.IN­TER_AREA)
        else:
            image_norm = cv2.resize(image, size,interpolation=cv2.IN­TER_CUBIC)
            
        images_norm.append(i­mage_norm)
        
    return images_norm



def normalize_faces(imag­e, faces_coord):

    faces = cut_faces(image, faces_coord)
    faces = resize(faces)
    
    return faces

for image in images:
    detector = FaceDetector("haarca­scade_frontalface_de­fault.xml")
    faces_coord = detector.detect(imag­e, True)
    faces = normalize_faces(imag­e ,faces_coord)
    for i, face in enumerate(faces):
        cv2.imwrite('%s.jpeg­' % (count), faces[i])
        count += 1




























