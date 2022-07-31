from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


digits = load_digits()

digits.data.shape
digits.target.shape

plt.figure(figsize = (20,4))
for index, (image, label) in enumerate(zip(digits.data[0:6], digits.target[0:6])):
    plt.subplot(1,5, index + 1)
    plt.imshow(np.reshape(image, (8,8)), cmap =plt.cm.gray)
    plt.title("Training: %i\n" % label, fontsize = 10)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target, test_size = 0.23, random_state = 2)

X_train.shape

y_train.shape

from sklearn.linear_model import LogisticRegression

log = LogisticRegression()

log.fit(X_train, y_train)

log.predict((X_test[0]).reshape(1,-1))

log.predict(X_test[0:20])

pred = log.predict(X_test)

score = log.score(X_test, y_test)

cm = metrics.confusion_matrix(y_test, pred)



index = 0
classifiedIndex = []
for predict, actual in zip(pred, y_test):
    if predict == actual:
        classifiedIndex.append(index)
        
    index+=1
    
plt.figure(figsize= (20, 3))
for plotIndex, wrong in enumerate(classifiedIndex[0:4]):
    plt.subplot(1,4, plotIndex +1)
    plt.imshow(np.reshape(X_test[wrong],(8,8)),cmap =plt.cm.gray )
    plt.title("predicted: {}, Actual: {} ".format(pred[wrong], y_test[wrong]), fontsize= 20)






