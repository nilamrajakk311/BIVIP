import numpy as np
import cv2
import math
pencasc_vert = 'pen_vertical_classifier.xml'
penCascade_vert = cv2.CascadeClassifier(pencasc_vert)
video_capture = cv2.VideoCapture('out.avi')
ret = True
def Hough_Transform(x,y,w,h,image):
    height, width = image.shape[:2]
    if (x>100) and (x+w+100<width) and (y>20) and (y+h+20<height):
        cropped_image = image[y-20:y+h+20,x-100:x+w+100]
    elif (x>50) and (x+w+50<width) and (y>10) and (y+h+10<height):
        cropped_image = image[y - 10:y + h + 10, x - 50:x + w + 50]
    elif (x > 100) and (x + w + 100 < width) and (y > 10) and (y + h + 10 < height):
        cropped_image = image[y - 10:y + h + 10, x - 100:x + w + 100]
    elif (x > 50) and (x + w + 50 < width) and (y > 20) and (y + h + 20 < height):
        cropped_image = image[y - 20:y + h + 20, x - 50:x + w + 50]

        #cv2.resize(image, (200,40), interpolation=cv2.INTER_AREA)
    else:
        cropped_image = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    edge_image = cv2.Canny(gray,10,50,apertureSize = 3)
    lines = cv2.HoughLines(edge_image,1,np.pi/180,200)
    if lines is not None:
        print('Line found!')
        for rho,theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))
            cv2.line(video_capture,(x1,y1),(x2,y2),(0,0,255),2)
            alpha = theta / (2 * np.pi) * 360
            if alpha >= 180:
                angle = 360 - alpha
            else:
                angle = alpha

            cv2.putText(image,str(angle),(10,25),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
while ret:
    ret, image = video_capture.read(0)
    pens_vert = penCascade_vert.detectMultiScale(
        image,
        scaleFactor = 1.3,
        minNeighbors = 20,

        minSize = (25, 80),
        flags = cv2.CASCADE_SCALE_IMAGE
    )
    for (x,y,w,h) in pens_vert:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    if ret == False:
        break
    cv2.imshow("window", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
video_capture.release()
#out.release()
cv2.destroyAllWindows()