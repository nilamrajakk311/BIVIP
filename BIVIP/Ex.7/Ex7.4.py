import numpy as np
import cv2, time
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import scipy.ndimage as ndim


#Exercise7.4
PATH = '/Users/49174/PycharmProjects/BIVIP/'
FEATURES = 400
MASK = None
img = cv2.imread('Lena.png', 0)
sift = cv2.xfeatures2d.SIFT_create(FEATURES)
kp, des = sift.detectAndCompute(img, MASK)
print(len(des))
featImg = cv2.drawKeypoints(img, kp, None, (0, 0, 255), 4)
cv2.imwrite('LenaKP.png', featImg)


