import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("Iris.csv")

pd.set_option("display.max_columns", None)


cols = data.select_dtypes(include=[np.object])

cols.columns

data.Species.value_counts()

data.Species.replace({"Iris-setosa": 0, "Iris-versicolor":1, "Iris-virginica":2}, inplace = True)

data.isnull().sum()



from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters =3)

kmeans.fit(data)

kmeans.cluster_centers_

data.shape

np.unique(kmeans.labels_)

kmeans.fit(data).score(data)


nc = range(1, 8)
#print(nc)

kmean = [KMeans(n_clusters=i) for i in nc]

kmean

score = [kmean[i].fit(data).score(data) for i in range(len(kmean))]

print(score)

plt.plot(nc, score)
plt.grid()




aa = np.absolute(score)
print(aa)
plt.plot(nc, aa, marker = "*")
plt.grid()


cluster_df = pd.concat([data, pd.Series(kmeans.labels_)], axis =1)

cluster_df.head(12)


cluster_df.shape
cluster_df.rename(columns = {cluster_df.columns[6]: "cluster_no"}, inplace = True)

a=cluster_df.sort_values(["cluster_no"])



colormap = np.array(["Red", "Green", "Yellow"])
plt.scatter(data.SepalWidthCm, data.SepalLengthCm, c=colormap[kmeans.labels_])

plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

















































