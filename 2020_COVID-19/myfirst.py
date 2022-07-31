import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.feature_extraction.text import TfidfVectorize
from sklearn.metrics import  f1_score

import json

file_name = "./data/sentiment/books_small.json"

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        #r=review["reviewText"]
        #rev= review["overall"]
        reviews.append((review["reviewText"],review["overall"]))
        reviews[5]
       
        
        


class Category:
    ELECTRONICS = "ELECTRONICS"
    BOOKS = "BOOKS"
    CLOTHING = "CL0THING"
    GROCERY = "GROCERY"
    PATID = "PATID"
    
    
class Sentiment:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"
 
    
class Review:
    def __init__(self, category, text, score):
        self.category = category
        self.text = text
        self.score = score
        self.sentiment = self.get_sengtiment()
        
        
        
