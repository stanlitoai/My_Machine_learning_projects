import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter 


def rgb2hex(rgb):
    hex = "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return hex

print(rgb2hex([255, 0, 0]))


def plot_image_info(path, k=6):
    img_bar = cv2.imread(path)
    img_rgb = cv2.cvtColor(img_bar, cv2.COLOR_BGR2RGB)
    
    resize = cv2.resize(img_rgb, (64, 64), interpolation = cv2.INTER_AREA)
    
    img_list =  resize.reshape((resize.shape[0] * resize.shape[1], 3))
    
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(img_list)
    
    #count labels to find the most popular
    label_counts = Counter(labels)
    total_count = sum(label_counts.values())
    
    
    ##subset out the most popular centroid
    center_colors = list(clt.cluster_centers_)
    ordered_colors = [center_colors[i]/255 for i in label_counts.keys()]
    color_labels = [rgb2hex(ordered_colors[i]*255) for i in label_counts.keys()]
    
    
    
#    print(label_counts.values())
#    print(color_labels)
    
    
    #plot
    plt.figure(figsize=(14, 8))
    plt.subplot(221)
    plt.imshow(img_rgb)
    plt.axis("off")
    
    
    plt.subplot(222)
    plt.pie(label_counts.values(), labels=color_labels, colors=ordered_colors,
            startangle=90)
    plt.axis("equal")
    plt.show()
    
    
    
plot_image_info("bobcat.PNG")























image = cv2.imread("chi.jpg")
type(image)


image.shape

plt.imshow(image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image)


image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(image)





































color_1 = [225, 0, 0]
color_2 = [0, 225, 0]
color_3 = [0, 0, 225]
color_4 = [127, 127, 127]

plt.imshow(np.array([
        [color_1, color_2],
        [color_3, color_4],
        ]))



plt.imshow(np.array([
        [color_1, color_2],
        [color_3, color_4]
        ]))
                     











































