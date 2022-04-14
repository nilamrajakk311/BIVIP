import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
from pylab import *
from PIL import Image

# Exercise 5.4
import scipy.ndimage as ndim
path = '/Users/49174/PycharmProjects/BIVIP/'
def sobelFilter(npImg):
    Fm = np.zeros(npImg.shape)
    Fn = np.zeros(npImg.shape)
    ndim.sobel( input=npImg,axis=1,output=Fm)
    ndim.sobel( input=npImg,axis=0,output=Fn)
    return Fm, Fn
def gaussFilter(npImg,sigma):
    Fm = np.zeros(npImg.shape)
    Fn = np.zeros(npImg.shape)
    ndim.gaussian_filter(npImg, (sigma,sigma),(0,1), Fm)
    ndim.gaussian_filter(npImg, (sigma,sigma), (1,0), Fn)
    return Fm, Fn
def laplaceFilter( npImg ):
    Fmn = np.zeros(npImg.shape)
    ndim.laplace(npImg,output=Fmn)
    return Fmn
def laplaceGaussFilter( npImg, Dmn, sigma ):
    Fmn = np.zeros( npImg.shape)
    ndim.gaussian_laplace(npImg,sigma,output=Fmn)
    return Fmn
gray = np.array(Image.open(path+'Building.png').convert('L'),'uint8')
sigma = 5
Fm1, Fn1 = sobelFilter( gray )
Fm2, Fn2 = gaussFilter( gray, sigma )
Fmn1 = laplaceFilter( gray )
Fmn2 = laplaceGaussFilter( gray,np.asarray([[0,1,0],[1,-4,1],[0,1,0]]), sigma )
plt.subplot(3,2,1), plt.axis('off')
plt.imshow(Fm1,'gray'), plt.title('Sobel m')
plt.subplot(3,2,2), plt.axis('off')
plt.imshow(Fn1,'gray'), plt.title('Sobel n')
plt.subplot(3,2,3), plt.axis('off')
plt.imshow(Fm2,'gray'), plt.title('Gauss m')
plt.subplot(3,2,4), plt.axis('off')
plt.imshow(Fn2,'gray'), plt.title('Gauss n')
plt.subplot(3,2,5), plt.axis('off')
plt.imshow(Fmn1,'gray'), plt.title('Laplace mn')
plt.subplot(3,2,6), plt.axis('off')
plt.imshow(Fmn2,'gray'), plt.title('Laplace Gauss mn')
plt.show()