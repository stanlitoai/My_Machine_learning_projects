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

data1= np.array([[random.randint(1, 400) for i in range(2)] for j in range(50)],dtype = np.float64)

data2= np.array([[random.randint(300, 700) for i in range(2)] for j in range(50)],dtype = np.float64)

data3= np.array([[random.randint(600, 900) for i in range(2)] for j in range(50)],dtype = np.float64)

data = np.append(np.append(data1, data2, axis=0), data3, axis =0)

fig, ax = plt.subplots(figsize = (12, 8))
plt.scatter(data[:,0], data[:, 1], s = 200)


labels1 = np.array([0 for i in range(50)])

labels2 = np.array([1 for i in range(50)])

labels3 = np.array([2 for i in range(50)])

labels = np.append(np.append(labels1, labels2, axis = 0), labels3, axis=0)

df = pd.DataFrame({"data_x": data[:,0], "data_y": data[:,1], "labels": labels})

df.sample(10)

colors = ["green", "blue","purple"]

plt.figure(figsize = (12,8))

plt.scatter(df["data_x"], df["data_y"], c=df["labels"], s = 200, cmap = matplotlib.colors.ListedColormap(colors))

kmean = KMeans(n_clusters= 3, max_iter = 10000)

kmean.fit(data)

a=kmean.labels_

centroids = kmean.cluster_centers_

fig, ax = plt.subplots(figsize = (12, 8))

plt.scatter(centroids[:,0], centroids[:,1], c="r", s= 250, marker = "s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] +7, centroids[i][1] +7), fontsize = 20)


print("Homogeneity_score: ", metrics.homogeneity_score(labels, kmean.labels_))


print("Completeness_score: ", metrics.completeness_score(labels, kmean.labels_))

print("V_measure_score: ", metrics.v_measure_score(labels, kmean.labels_))

print("Adjusted_rand_score: ", metrics.adjusted_rand_score(labels, kmean.labels_))

print("Adjusted_mutual_info_score: ", metrics.adjusted_mutual_info_score(labels, kmean.labels_))

print("Silhouette_score: ", metrics.silhouette_score(data, kmean.labels_))






colors = ["green", "blue","purple"]

plt.figure(figsize = (12,8))

plt.scatter(df["data_x"], df["data_y"], c=df["labels"], s = 200, cmap = matplotlib.colors.ListedColormap(colors))

plt.scatter(centroids[:,0], centroids[:,1], c="r", s= 250, marker = "s")

for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] +7, centroids[i][1] +7), fontsize = 30)


data_test = np.array([[442., 621.],
                     [50., 153.],
                     [333., 373.],
                     [835., 816.]])

label_pred = kmean.predict(data_test)








colors = ["green", "blue","purple"]

plt.figure(figsize = (12,8))

plt.scatter(df["data_x"], df["data_y"], c=df["labels"], s = 200, cmap = matplotlib.colors.ListedColormap(colors))

plt.scatter(data_test[:,0],data_test[:,1], c="orange", s= 300, marker="^")

for i in range(len(label_pred)):
    plt.annotate(label_pred[i], (data_test[i][0] +7, data_test[i][0] +7), fontsize=30)


plt.scatter(centroids[:,0], centroids[:,1], c="r", s= 250, marker = "s")
    
for i in range(len(centroids)):
    plt.annotate(i, (centroids[i][0] +7, centroids[i][1] +7), fontsize = 30)



