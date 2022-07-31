from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as  np
import pandas as pd
from sklearn.cluster import KMeans


iris = datasets.load_iris()

print(iris.DESCR)
iris.target_names
len(a=iris.data)

X = iris.data[:, :2], iris.data[:, :1], iris.data[:, :0]
y = iris.target

plt.scatter(X[:, 0], X[:, 1], c = y)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")



km = KMeans(n_clusters= 3, n_jobs = 4, random_state = 21) 
km.fit(X)




#PLOTTING THE GRAPHY
km_labels = km.labels_

fig, axes = plt.subplots(1, 2, figsize = (11, 5))
axes[0].scatter(X[:, 0], X[:, 1], c = km_labels)
axes[1].scatter(X[:, 0], X[:, 1], c = y)
axes[0].set_xlabel("Sepal length")
axes[1].set_ylabel("Sepal width")
axes[0].set_xlabel("Sepal length")
axes[1].set_ylabel("Sepal width")
axes[0].set_title("predicted")
axes[1].set_title("Actual")





























































