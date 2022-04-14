import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image

# Exercise 5.3
import scipy.ndimage as ndim
path = '/Users/49174/PycharmProjects/BIVIP/'
def sobelFilter1( npImg ):
    Fm = np.zeros(npImg.shape)
    Fn = np.zeros(npImg.shape)
    ndim.sobel(input=npImg,axis=1,output=Fm)
    ndim.sobel(input=npImg,axis=0, output=Fn)
    magnitude = np.sqrt(Fm ** 2 + Fn ** 2)
    phase = np.arctan2(Fm,Fn)
    return Fm, Fn, magnitude, phase
def sobelFilter2( npImg,Dm,Dn ):
    Fm = np.zeros(npImg.shape)
    Fn = np.zeros(npImg.shape)
    ndim.correlate(input=npImg,weights=Dm,output=Fm)
    ndim.correlate(input=npImg,weights=Dn,output=Fn)
    magnitude = np.sqrt(Fm ** 2 + Fn ** 2)
    phase = np.arctan2(Fm,Fn)
    return Fm, Fn, magnitude, phase
Dm = np.asarray([[-1,0,1],[-2,0,2],[-1,0,1]])
Dn = np.asarray([[-1,-2,-1],[0,0,0],[1,2,1]])
Ddm = np.asarray([[-2,-1,0],[-1,0,1],[0,1,2]])
Ddn = np.asarray([[0,-1,-2],[1,0,-1],[2,1,0]])
gray = np.array(Image.open(path+'chessfieldfoto.png').convert('L'),'uint8')
Fm1,Fn1,magnitude1,phase1 = sobelFilter1(gray)
Fm2,Fn2,magnitude2,phase2 = sobelFilter2(gray, Dm,Dn)
Fm3,Fn3,magnitude3,phase3 = sobelFilter2(gray, Ddm,Ddn)
plt.subplot(5,3,1), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Sobel 1')
plt.subplot(5,3,2), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Sobel 2')
plt.subplot(5,3,3), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Sobel 3')
plt.subplot(5,3,4), plt.axis('off')
plt.imshow(Fm1,'gray')
plt.subplot(5,3,5), plt.axis('off')
plt.imshow(Fm2,'gray'),plt.title('Fm Vertical edges')
plt.subplot(5,3,6), plt.axis('off')
plt.imshow(Fm3,'gray')
plt.subplot(5,3,7), plt.axis('off')
plt.imshow(Fn1,'gray')
plt.subplot(5,3,8), plt.axis('off')
plt.imshow(Fn2,'gray'), plt.title('Fn Horizontal edges')
plt.subplot(5,3,9), plt.axis('off')
plt.imshow(Fn3,'gray')
plt.subplot(5,3,10), plt.axis('off')
plt.imshow(magnitude1,'gray')
plt.subplot(5,3,11), plt.axis('off')
plt.imshow(magnitude2,'gray'),plt.title('Absolute Value')
plt.subplot(5,3,12), plt.axis('off')
plt.imshow(magnitude3,'gray')
plt.subplot(5,3,13), plt.axis('off')
plt.imshow(phase1,'gray')
plt.subplot(5,3,14), plt.axis('off')
plt.imshow(phase2,'gray'),plt.title('Phase')
plt.subplot(5,3,15), plt.axis('off')
plt.imshow(phase3,'gray')
plt.show()