
# ###################################################################################
# Exercise 9.7

# library imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

##################################################################################################
def grabCut(frame,x, y, w, h):

    #read the image
    #frame = cv2.imread(frame)
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
    return frame

#################################################################################################
def addAlphaChannel(img):
    # read the GrabCut – file
    #img = cv2.imread(img)
        # color space transform
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # convert GrabCut to image with alpha channel
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
        # set the alpha channel to transparent by use of a mask
        # “axis” defines the way of processing rows or columns
    img[np.all(img == [0, 0, 0, 255], axis=2)] = [0, 0, 0, 0]
    return img

######################################################################################
def pasteImage(alpha_grabcut, background_frame, w, h):

    # read images
    ag_img = alpha_grabcut
    b_img = background_frame
    # set x & y coordinates
    x = 0
    y = 0
    # verification of forwarded data
    if ag_img is not None:
        print("ag_img: " + str(ag_img.shape))
    else:
        print("no ag_img")
    if b_img is not None:
        print("b_img: " + str(b_img.shape))
    else:
        print("no b_img")

    # scale alpha channel, create FG mask and determine BG mask
    alpha_ag = ag_img[:, :,  3] / 255.0
    alpha_b = 1.0 - alpha_ag

    # actual image pasting
    for c in range(0, 3):
        b_img[y:y + h, x:x + w, c] = (alpha_ag * ag_img[:, :, c] + alpha_b * b_img[y:y + h, x:x + w, c])
    return b_img

def MLBackgroundReplacement(background_video, foreground_video, classifier):

     # create video capture objects
    cap = cv2.VideoCapture(foreground_video)
    background_cap = cv2.VideoCapture(background_video)
    # load pretrained XML-weight-files for Haar Cascade classifier
    object_classifier = cv2.CascadeClassifier(classifier)
    # initialize frame counter
    frame_counter = 0
    # initialize video writer codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # get width & height of VideoCapture object
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # initialize video writer object
    out = cv2.VideoWriter("out.mp4", fourcc, 10.0, (width,height))
    # limit iterations/ maximum frames processed
    max_frames= min(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), int(background_cap.get(cv2.CAP_PROP_FRAME_COUNT)))

    # process every frame of the video stream
    while frame_counter < max_frames:
        # capture a single frame for the current iteration
        ret, frame =  cap.read()
        # load correspondent background frame from video capture object
        success, background_frame = background_cap.read()
        print("frame" + str(frame_counter))
        print(background_frame.shape)
        # convert the image to grayscale
        gray =  cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
        # detect objects (might be more than one) in the current frame
        detected_objects = object_classifier.detectMultiScale(gray, 1.1,4)

        if detected_objects is not None:
            for (x,y,w,h) in detected_objects:

                    # crop the objects out of the frame
                    grabcut_object= grabCut(frame, x,y, w, h)
                    # convert to grabcut with alpha channel
                    alpha_grabcut = addAlphaChannel(grabcut_object)
                    # paste alpha_grabcut onto appropriate background frame
                    background_frame = pasteImage(alpha_grabcut, background_frame,width,height)



        # save composite to video writer object
        out.write(background_frame)
        # increase frame_counter
        frame_counter +=1
    # release video capture and video writer objects
    cap.release()
    background_cap.release()
    out.release()

    # close all windows
    cv2.destroyAllWindows()
#cap = cv2.VideoCapture('johnny.avi')
#background =  cv2.VideoCapture('Beach.mp4')
#classifier= 'haarcascade_frontalface_deafault.xml'
MLBackgroundReplacement("Beach.mp4","Johnny.avi", 'haarcascade_frontalface_default.xml')
#MLBackgroundReplacement(cap,background,classifier)