# ###################################################################################
# Exercise 9.3

#library imports
import numpy as np
import cv2
import math
#initialization of video capture object; set webcam as input
cap = cv2.VideoCapture('johnny.avi')
#create background subtractor and kernel, if necessary
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
#write while-loop, that is executed if the VideoCapture object is opened
cap = cv2.VideoCapture('johnny.avi')
while(cap.isOpened()):

    #read cap and apply the background subtractor on the current frame
    ret, image = cap.read()
    fgmask = fgbg.apply(image)
    #only the version with kernel needs and additional morphology
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    #show the resulting frame and add an option to quit by pressing a key
    cv2.imshow('windows', fgmask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()

