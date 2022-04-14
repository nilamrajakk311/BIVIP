# ###################################################################################
# Experiment 8 Code
# ###################################################################################

# ###################################################################################
# Exercise 8.1
import cv2 

video_capture = cv2.VideoCapture(0)
ret = True

while ret:
  ret, gray = video_capture.read(0)
  if ret == False:
    break
  cv2.imshow('windows',gray)
  
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

video_capture.release()
cv2.destroyAllWindows()


# ###################################################################################
# Exercise 8.2
import cv2
import sys
import time
video_capture = cv2.VideoCapture(0)
video_capture.set(3,1280)                                                  
video_capture.set(4,720)                                                  
video_capture.set(5, 10)                                                  
ret = True
no_frames = 200
cnt = 0
start = time.time()
while ret:
    cnt = cnt + 1
    if cnt==no_frames:
        end = time.time()                                                
        seconds = end-start
        print("Time taken = {0} sec".format(seconds))
        fps = no_frames/seconds
        print("Estimated FPS = {0} ".format(fps))
        cnt = 0
        start = time.time()
    # Capture frame-by-frame
    ret, gray = video_capture.read(0)
    if ret == False:
        break
    cv2.imshow("window", gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


# ###################################################################################
#Exercise 8.3
import cv2
import sys
pencasc_vert = 'pen_vertical_classifier.xml'
pencasc_hor = 'pen_horizontal_classifier.xml'
penCascade_vert = cv2.CascadeClassifier(pencasc_vert)
penCascade_hor = cv2.CascadeClassifier(pencasc_hor)
video_capture = cv2.VideoCapture('out.avi')
#video_capture = cv2.VideoCapture(0)
ret = True
while ret:
    # Capture frame-by-frame
    ver_flag = False                                                         
    ret, gray = video_capture.read(0)
    if ret == False:
        break
    pens_vert = penCascade_vert.detectMultiScale(
        gray,
        scaleFactor=1.7,
        minNeighbors=25,
        minSize=(25, 80),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    # Draw a rectangle around the objects
    for (x, y, w, h) in pens_vert:
        cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
        #ver_flag = True
        
    pens_hor = penCascade_hor.detectMultiScale(
      gray,
      scaleFactor=1.8,
      minNeighbors=30,
      minSize=(80, 35),
      flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (p, q, r, s) in pens_hor:
        if ver_flag != True:        
            cv2.rectangle(gray,(p,q),(p+r,q+s),(0,255,0),2)
    cv2.imshow("window", gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
