# ###################################################################################
# Experiment 9 Code
# ###################################################################################

# ###################################################################################
# Exercise 9.1

#import libraries
import numpy as np
import cv2
import matplotlib
import math
import matplotlib.pyplot as plt
matplotlib.use('tkagg')
#import args
from pylab import *
from PIL import Image

#create VideoCapture object
cap = cv2.VideoCapture('johnny.avi')
#use pre-trained Haar Cascade XML classifier
haarcascade = 'haarcascade_frontalface_default.xml'
object_classifier = cv2.CascadeClassifier(haarcascade)
#read until video is completed/ process every frame
while True:
    #capture a single frame for the current iteration
    ret, frame = cap.read()
    #convert frame into gray scale 
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    #detect objects in the frame
    detected_objects = haarcascade.detectMultiScale(frame, 1.1, 4)
   
    #to draw rectangle on each detected object in the current frame
    for (x, y, w, h) in haarcascade:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    #display the resulting frame
    if ret == False:
        break
    cv2.imshow("window", frame)
    #press Q on keyboard to exit
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
    
#release VideoCapture object
cap.release()
#close all windows
cv2.destroyAllWindows()
# ###################################################################################
# Exercise 9.2

#library import and initialization of video capture object

#define the codec (fourcc) and create video writer object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("out.mp4", fourcc, 10.0, (700,466))

while True:
    #TODO
    if ret==True:
        #frame = cv2.flip(frame,0)
        #write and show the (flipped) frame
        #TODO
        if #TODO:
            break
    else:
        break

#release everything if job is finished
#TODO



