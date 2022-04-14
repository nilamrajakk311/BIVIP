#4.3
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from PIL import Image
import cv2


LenaRGB = np.array(Image.open('lena.png'), dtype=np.uint8)
img = npimg.imread('lena.png')
def loadImage2Array(fileName):
    try:
        Lena2RGB = np.array(Image.open(fileName), dtype=np.uint8)
        return Lena2RGB
    except:
        return False
arr = loadImage2Array('lena.png')
print(arr)

def saveArray2GrayImg(npArrayImg, fileName):
    assert npArrayImg.dtype == "uint8"

    if npArrayImg.dtype != 'uint8':
        # npArrayImg = np.array(npArrayImg, dtype='uint8') # Option 1
        npArrayImg = npArrayImg.astype(np.dtype('uint8'))  # Option 2
    return npimg.imsave(fileName, npArrayImg, cmap='gray')


saveArray2GrayImg(arr,'lena1.png')
def convertRGB_equal(imgRGB):

    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = (R+G+B)/3
    plt.imshow(imgGray,cmap='gray')
    plt.show()
gray1 = convertRGB_equal(LenaRGB)

def convertRGB_weighted1(imgRGB):
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray,cmap='gray')
    plt.show()
gray2 = convertRGB_weighted1(LenaRGB)


def convertRGB_weighted2(imgRGB):
    return np.dot(LenaRGB[...,:3], [0.299, 0.587, 0.144])

gray3 = convertRGB_weighted2(img)
plt.imshow(gray3,cmap='gray')
plt.show()
#plt.savefig('grey3.png')

def plothistogram(imgGray):
    imghistogram, bins = np.histogram(imgGray.flatten(),256)
    plt.subplot(121),plt.title('Histogram 1')#plt.xlim(0,255),plt.ylim(0,500)
    plt.plot(imghistogram)
    plt.subplot(122),plt.title('Histogram 2')#plt.xlim(0,255),plt.ylim(0,1500)
    plt.hist(imgGray.flatten(),bins=100)
    plt.show()
plothistogram(gray3)