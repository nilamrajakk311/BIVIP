# ###################################################################################
# Exercise 9.5

# library imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
def addAlphaChannel(img):
        # read the GrabCut – file
        img = cv2.imread('johnny_grabcut.png')
        # color space transform
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # convert GrabCut to image with alpha channel
        img=  cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
        # set the alpha channel to transparent by use of a mask
        # “axis” defines the way of processing rows or columns
        img[np.all(img == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]

        # write the img
        cv2.imwrite('alpha_johhny_grabcut'+'.png',img)
        plt.imshow(img),plt.show()
        #cv2.imshow('window',img)
t1 = time.time()
addAlphaChannel('johnny_grabcut.png')
t2 = time.time()
t = t2 -t1
print(t)
