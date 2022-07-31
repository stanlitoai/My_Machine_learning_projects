##OCR digit recognition == optimal character recognition

import os
import gzip
import tensorflow as tf
import pandas as pd
import numpy as np
import csv
import matplotlib
import matplotlib.pyplot as plt

from six.moves import urllib

##downloading the file
URL_PATH = "http://ai.stanford.edu/~btaskar/ocr/letter.data.gz"
downloaded_filename= "letter.data.gz"

##checking and downloading the files
def download_data():
    if not os.path.exists(downloaded_filename):
        filename,_ = urllib.request.urlretrieve(URL_PATH, downloaded_filename)
        
    print("Found and verified file from this path: ", URL_PATH)
    print("Downloaded file: ", downloaded_filename)
    
download_data() 

##Now reading the fiole we downloaded


def read_lines():
    
    with gzip.open(downloaded_filename, "rt") as f:
        reader = csv.reader(f, delimiter="\t")
        lines = list(reader)
        
        return lines
    
lines = read_lines()

len(lines)
lines[1][:15]

##GETTING THE FEATURE LABELS\

def get_features_labels(lines):
    lines = sorted(lines, key=lambda x: int(x[0]))
    
    data, target = [], []
    
    next_id = -1
    
    word = []
    word_pixels = []
    
    for line in lines:
        next_id = int(line[2])
        
        pixels = np.array([int(x) for x in line[6:134]])
        pixels = pixels.reshape((16, 8))
        
        word_pixels.append(pixels)
        
        word.append(line[1])
        
        if next_id == -1:
            
            data.append(word_pixels)
            target.append(word)
            
            word = []
            word_pixels = []
            
    return data, target


data, target = get_features_labels(lines)

len(data), len(target)

###
np.zeros((16,8))

##

####GETTING THE FEATURE LABEL FOR PADDING 

def pad_features_labels(data, target):
    
    max_length = max(len(x) for x in target)
    
    padding = np.zeros((16,8))
    
    # pad the image data with the empty str images
    data = [x + ([padding] * (max_length - len(x))) for x in data]
    
    #pad the words with empty str characters
    target = [x + ([""] * (max_length - len(x))) for x in target]
    
    return np.array(data), np.array(target)


padded_data, padded_target = pad_features_labels(data, target)

len(padded_data), len(padded_target)

padded_target[200:210]

word_length = len(padded_target[0])

padded_data.shape, padded_target.shape


padded_data.shape[:2] + (-1,)

reshaped_data = padded_data.reshape(padded_data.shape[:2] + (-1,))

reshaped_data.shape

padded_target.shape

padded_target.shape + (26,)

##lets use to hold the one_hot_encoding of each of the characters
one_hot_target = np.zeros(padded_target.shape + (26,))

##using ndenumerate the iterate every word and every target in our word

for index, letter in np.ndenumerate(padded_target):
    
    #now assigning the character like a = 1, b=2.......(in lower case)
    if letter:
        one_hot_target[index][ord(letter) - ord("a")] = 1
        

one_hot_target[0][0]

shuffled_indices = np.random.permutation(len(reshaped_data))

shuffled_data = reshaped_data[shuffled_indices]
shuffled_target = one_hot_target[shuffled_indices]


split = int(0.66 * len(shuffled_data))

train_data = shuffled_data[:split]
train_target = shuffled_target[:split]


test_data = shuffled_data[split:]
test_target = shuffled_target[split:]
 



























 























































