import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV, train_test_split



dataset = pd.read_csv("spam.csv")
dataset.describe()
dataset.drop_duplicates(inplace=True)
dataset.isnull().sum()

X = dataset["Label"]
y = dataset["EmailText"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train.shape, X_test.shape,y_train.shape,y_test.shape

cv = CountVectorizer()
features = cv.fit_transform(X_train)

features_test = cv.fit_transform(X_test)

features = toa


print(cv.get_feature_names)

tuned_para = {"kernel": ["linear","rbf"], "gamma":[1e-3,1e-4],
              "C":[1, 10, 100, 1000]}

model = GridSearchCV(svm.SVC(), tuned_para)

model.fit(features.shape, y_train)

print(model.best_params_)


print(model.score(features_test.shape, y_test))














