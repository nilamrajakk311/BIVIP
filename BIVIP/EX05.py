import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image

# Exercise 5.1
import scipy.ndimage as ndim
path = '/Users/49174/PycharmProjects/BIVIP/'
def edgeDetector1D( npImg, Dm, Dn ):
    Fm = np.zeros(npImg.shape)
    Fn = np.zeros(npImg.shape)
    ndim.correlate1d(npImg,Dm,1,Fm)
    ndim.correlate1d(npImg,Dn,0,Fn)
    magnitude = np.sqrt(Fm**2+Fn**2)
    phase = np.arctan2(Fm,Fn)
    return Fm, Fn, magnitude, phase
gray = np.array( Image.open( path + 'Chessboard.png').convert('L'),'uint8')
Dm1 = np.asarray([-1,1,0])
Dn1 = np.asarray([-1,1,0])
Dm2 = np.asarray([-1,0,1])
Dn2 = np.asarray([-1,0,1])
Dm3 = np.asarray([0,-1,1])
Dn3 = np.asarray([0,-1,1])
Fm1, Fn1, magnitude1, phase1 = edgeDetector1D(gray, Dm1, Dn1)
Fm2, Fn2, magnitude2, phase2 = edgeDetector1D(gray, Dm2, Dn2)
Fm3, Fn3, magnitude3, phase3 = edgeDetector1D(gray, Dm3, Dn3)


plt.subplot(5,3,1), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Gradient 1')
plt.subplot(5,3,2), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Gradient 2')
plt.subplot(5,3,3), plt.axis('off')
plt.imshow(gray,'gray'), plt.title('Gradient 3')

#vertical edges
plt.subplot(5,3,4), plt.axis('off')
plt.imshow(Fm1,'gray')
plt.subplot(5,3,7), plt.axis('off')
plt.imshow(Fn1,'gray')
plt.subplot(5,3,10), plt.axis('off')
plt.imshow(magnitude1,'gray')
plt.subplot(5,3,13), plt.axis('off')
plt.imshow(phase1,'gray')

plt.subplot(5,3,5), plt.axis('off')
plt.imshow(Fm2,'gray'),plt.title('Fm Vertical edges')
plt.subplot(5,3,8), plt.axis('off')
plt.imshow(Fn2,'gray'), plt.title('Fn Horizontal edges')
plt.subplot(5,3,11), plt.axis('off')
plt.imshow(magnitude2,'gray'),plt.title('Absolute Value')
plt.subplot(5,3,6), plt.axis('off')
plt.imshow(Fm3,'gray')

#horizontal edges


plt.subplot(5,3,9), plt.axis('off')
plt.imshow(Fn3,'gray')

#magnitude


plt.subplot(5,3,12), plt.axis('off')
plt.imshow(magnitude3,'gray')

#phase

plt.subplot(5,3,14), plt.axis('off')
plt.imshow(phase2,'gray'),plt.title('Phase')
plt.subplot(5,3,15), plt.axis('off')
plt.imshow(phase3,'gray')
plt.show()