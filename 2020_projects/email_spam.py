import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from nltk.corpus import stopwords
import strings
from sklearn.feature_extraction.text import CountVectorizer


dataset = pd.read_csv("spam.csv")
dataset.describe()
dataset.drop_duplicates(inplace=True)
dataset.isnull().sum()

def process_text(text):
    
    #1 remove the punctuation
    #2 remove the stopwords
    #3 return a list of clean text words
    
    #1
    nopunc = [char for char in text if char not in string.punctuation]
    nopunc = ''.join(nopunc)
    
    #2
    clean_words = [word for word in nopunc.split() if word.lower() not in stopwords.words("english")]
    
    #3
    return clean_words


#show the tokenization( a list  ofmtoke also called lemmas)

dataset["EmailText"].head().apply(process_text)


cv = CountVectorizer(analyzer=process_text)
messages = cv.fit_transform(dataset["EmailText"])



X_train, X_test, y_train, y_test = train_test_split(messages, dataset["Label"], test_size=0.2, random_state=0)

X_train.shape, X_test.shape,y_train.shape,y_test.shape

#Create and train the naive bayes classifier

from sklearn.naive_bayes import MultinomialNB, GaussianNB

classifier = MultinomialNB()

classifier.fit(X_train, y_train)

#Print the predictions
print(classifier.predict(X_train))

#Print actual values
print(y_train.values)

#Evaluate the model on the training dataset

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

pred = classifier.predict(X_train)
print(classification_report(y_train, pred))

print("Confusion matrix: \n", confusion_matrix(y_train, pred))

print("Accuracy: ", accuracy_score(y_train, pred))


#Print the predictions
print(classifier.predict(X_test))

#Print actual values
print(y_test.values)


pred = classifier.predict(X_test)
print(classification_report(y_test, pred))

print("Confusion matrix: \n", confusion_matrix(y_test, pred))

print("Accuracy: ", accuracy_score(y_test, pred))




































