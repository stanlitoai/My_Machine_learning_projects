from sklearn.model_selection import train_test_split



class Sentiment:
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class Review:
    def __init__(self,text, score):
        self.text = text
        self.score = score
        self.sentiment = self.get_sentiment()
        
        
    def get_sentiment(self):
        if self.score <= 2:
            return Sentiment.NEGATIVE
        elif self.score == 3:
            return Sentiment.NEUTRAL
        else:
            return Sentiment.POSITIVE



import json

file_name = "./data/sentiment/books_small.json"

reviews = []
with open(file_name) as f:
    for line in f:
        review = json.loads(line)
        #r=review["reviewText"]
        #rev= review["overall"]
        reviews.append(Review(review["reviewText"],review["overall"]))
        
        
        

        
#Preparing your  data       
        
training,test = train_test_split(reviews, test_size = 0.33, random_state = 42)
test[5].score
#training[5].score

train_x = [x.text for x in training]
train_y = [x.sentiment for x in test]

test_x = [x.text for x in test]
test_y = [x.sentiment for x in test]


train_x[0]
train_y[0]



#Bag of words vectorization

from sklearn.feature_extraction.text import CountVectorizer

vec = CountVectorizer()

train_x_vec = vec.fit_transform(train_x)
test_x_vec = vec.transform(test_x)

train_x[0]
train_x_vec[0].toarray()



#Classification
#Linear SVM

from sklearn import svm

clf_svm =  svm.SVC(kernel = "linear")
clf_svm.fit(train_x_vec, train_y)

test_x[0]

clf_svm.predict(test_x_vec[0])


#Decision tree

from sklearn.tree import DecisionTreeClassifier

clf_dec = DecisionTreeClassifier()
clf_dec.fit(train_x_vec, train_y)














































































        
        
        
        
        
        
        
        
        
        
        
        
        
        
  
        
reviews[113].text        
reviews[113].sentiment
       
        