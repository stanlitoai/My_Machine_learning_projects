import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io


img = cv2.imread("chi.jpg")
plt.imshow(img)
plt.show()


red = img[:, :, 1]
green = img[:, :, 1]
blue = img[:, :, 1]


plt.imshow(red, cmap = "Reds")
plt.show()

plt.imshow(green, cmap = "Greens")
plt.show()

plt.imshow(blue, cmap = "Blues")
plt.show()


def filtering(img, f=3):
    
    #Dimension from the input shape
    (rows, col, channels) = img.shape
    
    #Initialize hyper parameters
    stride = 2
    
    #Dimension of the input
    n_rows = int(1 + (rows - f)/ stride)
    n_col = int(1 + (col - f)/ stride)
    n_channels = channels
    
    
    #Initialize output matrix A
    n_img = np.zeros((n_rows, n_col, n_channels))
    
    
    # Iterate through img
    for h in range(n_rows):
        for w in range(n_col):
            for c in range(n_channels):
                
                vert_start = h*stride
                vert_end = vert_start + f
                horiz_start = w*stride
                horiz_end = horiz_start + f
                
                
                #extracting the slice we are dealing with
                n_slice = img[vert_start:vert_end, horiz_start:horiz_end, c]
                
                #complete the flattering operation on the slice
                n_img[h, w, c] = np.mean(n_slice, dtype=int)
                
    return n_img



A = filtering(img)

print(A.shape)
plt.imshow(filtering(img, f=11))
plt.show()
    
    


#CONVOLUTION NETWORK


    
    
    
    
    
    
    
    
    
    

































