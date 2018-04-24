from __future__ import division, print_function, absolute_import
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def load_data():
    X = []
    Y = []
    imagepaths = open('Dataset/dcnninname.txt')
    for line in imagepaths:
        #imagepath = 'Dataset/DEBUG/A0.png'
        imagepath = line.strip()
        image = plt.imread(imagepath)
        image_arr = np.array(image)

        #print(image_arr.shape)
        X.append(image_arr)
        Y.append(imagepath[-6])
        #print(imagepath,Y)

    return X,Y

if __name__ == '__main__':
    load_data()