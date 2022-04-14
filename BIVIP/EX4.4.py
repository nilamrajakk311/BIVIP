# Exercise 4.4
#!/usr/bin/python3
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from PIL import Image
from pylab import *
from scipy.stats import norm
import numpy as np
path = 'C:/Users/49174/PycharmProjects/BIVIP/'
def normalCDF():
    x = np.linspace(-4, 4, 50)
    y = [norm.cdf(x[i]) for i in range(x.size)]
    xval = np.linspace(-4,4,200)
    yval = np.interp(xval, x, y)
    return x, y, xval, yval
def convertRGB2Gray( imgName ):
    return np.array(Image.open(path + imgName).convert('L'))
def CumSum( nparray ):
    cmsum = nparray.copy()
    return np.cumsum(cmsum)
def histogramEqualize( img ):
    imghistogram, bins = np.histogram(img.flatten(),256)
    CDF = CumSum(imghistogram)
    CDF = 255 * CDF / max(CDF)
    imgEqual = np.interp(img.flatten(),bins[:-1],CDF)
    return imgEqual.reshape(img.shape), CDF
img1 = convertRGB2Gray('lena.png')
img2, CDF = histogramEqualize(img1)
x, y, xval, yval = normalCDF()
plt.subplot(231),
plt.imshow(img1,cmap='gray'),plt.title('Before')
plt.subplot(232)
plt.plot(x, y, 'bo'), plt.plot(xval, yval, 'rx')
plt.title('Normal cdf'), plt.xlabel('x'), plt.ylabel('Probability')
plt.subplot(233),
plt.imshow(img2,cmap='gray'),plt.title('After')
plt.subplot(234),
plt.title('Histogram 1'), plt.xlim(0, 255)
plt.hist(img1.flatten(), 64)
plt.subplot(235), plt.title('Transformation')
plt.plot(range(CDF.size), CDF[:])
plt.subplot(236), plt.title('Histogram 2'), plt.xlim(0, 255)
plt.hist(img2.flatten(), 64)
plt.show()