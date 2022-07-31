import matplotlib.pyplot as plt
import numpy as np

from  sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples= 300, centers = 4, cluster_std = 0.60, random_state = 0)

plt.scatter(X[:, 0], X[:,1], s= 50)

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 4)
kmeans.fit(X)

y_kmeans = kmeans.predict(X)


from sklearn.datasets import load_sample_image
stan = load_sample_image("flower.jpg")
ax = plt.axes(xticks=[], yticks=[])
ax.imshow(stan);

#returnms the dimeansions of the array
stan.shape

#reshape the data to [n_samples x n_features], and rescale the colors so that they lie between 0 and 1
data = stan/ 255.0  #use 0...1 scale
data = data.reshape(427 * 640, 3)
data.shape

#visualize these pixels in this color space, using a subset of 10,000 pixels for efficiency

def plot_pixels(data, title, colors= None, N= 10000):
    if colors is None:
        colors =data
    
    #choose a random subset
    rng = np.random.RandomState(0)
    i=rng.permutation(data.shape[0])[:N]
    colors= colors[1]
    R,G,B = data[i].T
    
    
    fig, ax = plt.subplots(1,2, figsize=(16,6))
    ax[0]
    









