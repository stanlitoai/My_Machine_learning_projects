%matplotlib
import sklearn 
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
#import keras
import tensorflow


iris_df = pd.read_csv("iris.csv", skiprows=1, names= ["sepal-lenght",
                                                      "sepal-width",
                                                      "petal-length",
                                                      "petal-width",
                                                      "class"])
iris_df.head()

iris_df.shape

iris_df = iris_df.sample(frac=1).reset_index(drop= True)

from sklearn import preprocessing 

label_encoding=preprocessing.LabelEncoder()

iris_df["class"] = label_encoding.fit_transform(iris_df["class"].astype(str))

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(iris_df["sepal-lenght"], iris_df["sepal-width"], s = 250)

plt.xlabel("sepal-lenght")
plt.ylabel("sepal-width")
#plt.legend
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(iris_df["petal-width"], iris_df["petal-length"], s = 250)

plt.xlabel("petal-width")
plt.ylabel("petal-lenght")
#plt.legend
plt.show()

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(iris_df["sepal-lenght"], iris_df["petal-length"], s = 250)

plt.xlabel("sepal-length")
plt.ylabel("petal-length")
#plt.legend
plt.show()

iris_2d = iris_df[["sepal-lenght", "petal-length"]]

iris_2d.sample(5)

iris_2d = np.array(iris_2d)

kmeans_model= KMeans(n_clusters =3, max_iter = 1000)

kmeans_model.fit(iris_2d)

kmeans_model.labels_

centroids = kmeans_model.cluster_centers_

centroids

fig, ax = plt.subplots(figsize = (12, 8))

plt.scatter(centroids[:,0], centroids[:,1], c="r", s=250, marker="s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=30 )


iris_labels = iris_df["class"]


print("Homogeneity_score: ", metrics.homogeneity_score(iris_labels, kmeans_model.labels_))


print("Completeness_score: ", metrics.completeness_score(iris_labels, kmeans_model.labels_))

print("V_measure_score: ", metrics.v_measure_score(iris_labels, kmeans_model.labels_))

print("Adjusted_rand_score: ", metrics.adjusted_rand_score(iris_labels, kmeans_model.labels_))

print("Adjusted_mutual_info_score: ", metrics.adjusted_mutual_info_score(iris_labels, kmeans_model.labels_))

print("Silhouette_score: ", metrics.silhouette_score(iris_2d, kmeans_model.labels_))




colors = ["green", "blue","purple"]
fig, ax = plt.subplots(figsize = (12, 8))

plt.scatter(iris_df["sepal-lenght"], iris_df["petal-length"],c = iris_df["class"], s = 250, cmap = matplotlib.colors.ListedColormap(colors))

plt.scatter(centroids[:,0], centroids[:,1], c="r", s=250, marker="s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0], centroids[i][1]), fontsize=30 )




iris_features = iris_df.drop("class", axis =1)

iris_features.head()

iris_labels = iris_df["class"]

iris_labels.sample(5)

kmeans_model = KMeans(n_clusters=3)
kmeans_model.fit(iris_features)

kmeans_model.labels_

kmeans_model.cluster_centers_



print("Homogeneity_score: ", metrics.homogeneity_score(iris_labels, kmeans_model.labels_))


print("Completeness_score: ", metrics.completeness_score(iris_labels, kmeans_model.labels_))

print("V_measure_score: ", metrics.v_measure_score(iris_labels, kmeans_model.labels_))

print("Adjusted_rand_score: ", metrics.adjusted_rand_score(iris_labels, kmeans_model.labels_))

print("Adjusted_mutual_info_score: ", metrics.adjusted_mutual_info_score(iris_labels, kmeans_model.labels_))

print("Silhouette_score: ", metrics.silhouette_score(iris_features, kmeans_model.labels_))












































