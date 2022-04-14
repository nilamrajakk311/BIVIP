# ###################################################################################
# Exercise 9.4

#library imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def grabCut(frame,x, y, w, h):

    #read the image
    #frame = cv2.imread('johnny.png')
    frame = cv2.imread(frame)
    #create a mask out of zeros
    mask = np.zeros(frame.shape[:2], np.uint8)

    #create FG & BG model
    fgModel = np.zeros((1, 65), np.float64)
    bgModel = np.zeros((1, 65), np.float64)

    #compute rectangle coordinates from input
    #rect = (104,25,591,462)
    rect = (x, y, x + w, y + h)

    #perform OpenCV .grabCut()
    cv2.grabCut(frame,mask,rect,bgModel,fgModel,5,mode=cv2.GC_INIT_WITH_RECT)

    #convert the mask to binary for further operations
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')

    #this operation applies the mask onto our frame and creates a new axis
    frame = frame*mask2[:,:,np.newaxis]

    #write the frame to your directory
    cv2.imwrite('johnny_grabcut'+'.png',frame)
    plt.imshow(frame),plt.show()
    #cv2.imshow('windows',frame)

t1 = time.time()
grabCut("johnny.png", 75,25,455,465)
t2 = time.time()
t = t2-t1
print(t)
