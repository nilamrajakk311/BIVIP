import cv2
import numpy as np

cam = cv2.VideoCapture('out.avi')
i=0
while cam.isOpened():
    ret, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)
    #cv2.resize(gray, (1280, 720), fx=0, fy=0, interpolation=cv2.INTER_CUBIC)
    if i%50==0:
        cv2.imwrite('VideoCam' + str(i) + '.png', gray)
    i += 1
    if cv2.waitKey(1) & 0xFF == ord('s'):  # wait for 's' key to save and exit
        break
    cv2.imshow('Display', gray)
cam.release()
cv2.destroyAllWindows()