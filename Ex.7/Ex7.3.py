import numpy as np
import cv2, time
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import scipy.ndimage as ndim



# Exercise 7.3
import math
path = '/Users/49174/PycharmProjects/BIVIP/'
gray = cv2.imread('Lena.png', 0)
scaleLevels = 5
sigma = 1.6
k = math.sqrt(2)
scaleSpaceGauss = np.zeros((gray.shape[0], gray.shape[1], scaleLevels),dtype=np.uint8)
scaleSpaceDoG = np.zeros((gray.shape[0], gray.shape[1], scaleLevels-1),dtype=np.uint8)

for i in range(scaleLevels):
    kSigma = sigma*k**i
    scaleSpaceGauss[:,:,i] = cv2.GaussianBlur(gray, ksize=(0,0),sigmaX=kSigma,sigmaY=kSigma)
    cv2.imwrite(path + 'imgGauss_Oct1_' + str(i) + '.png',scaleSpaceGauss[:,:,i])
for i in range(scaleLevels-1):
    scaleSpaceDoG[:,:,i] = cv2.subtract(scaleSpaceGauss[:,:,i],scaleSpaceGauss[:,:,i+1])
    cv2.imwrite(path + 'imgDoG_Oct1_' + str(i+1) + '.png',scaleSpaceDoG[:,:,i])
nextOctgray = cv2.pyrDown(gray)
scaleSpaceGauss2 = np.zeros((nextOctgray.shape[0], nextOctgray.shape[1],scaleLevels), dtype=np.uint8)
scaleSpaceDoG2 = np.zeros((nextOctgray.shape[0], nextOctgray.shape[1],scaleLevels-1), dtype=np.uint8)
for i in range(scaleLevels):
    kSigma = sigma * k ** i
    scaleSpaceGauss2[:, :, i] = cv2.GaussianBlur(nextOctgray, ksize=(0, 0), sigmaX=kSigma, sigmaY=kSigma)
    cv2.imwrite(path + 'imgGauss_Oct2_' + str(i) + '.png', scaleSpaceGauss2[:, :, i])
for i in range(scaleLevels-1):
    scaleSpaceDoG2[:, :, i] =cv2.subtract(scaleSpaceGauss2[:,:,i],scaleSpaceGauss2[:,:,i+1])
    cv2.imwrite(path + 'imgDoG_Oct2_' + str(i + 1) + '.png', scaleSpaceDoG[:, :, i])
nextOctgray2 = cv2.pyrDown(nextOctgray)


