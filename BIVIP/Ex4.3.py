#4.3
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from PIL import Image
import time
LenaRGB = np.array(Image.open('lena.png'), dtype=np.uint8)
def loadImage2Array(fileName):
    try:
        Lena2RGB = np.array(Image.open(fileName), dtype=np.uint8)
        return Lena2RGB
    except:
        return False
arr = loadImage2Array('lena.png')
print(arr)

def saveArray2GrayImg(npArrayImg, fileName):
    #print(npArrayImg.dtype)
    assert npArrayImg.dtype == "uint8"
    #img = Image.open('lena.png')
    #imgGray = img.convert('L')
    #imgGray.save('lena.png')

    if npArrayImg.dtype != 'uint8':
        #npArrayImg = np.array(npArrayImg, dtype='uint8') # Option 1
        npArrayImg = npArrayImg.astype(np.dtype('uint8')) # Option 2
    return npimg.imsave(fileName, npArrayImg, cmap='gray')


#saveArray2GrayImg(arr,'lena1.png')

def convertRGB_equal(imgRGB):
    assert type(imgRGB) is np.ndarray

    #img = npimg.imread('lena.png')
    print(imgRGB)
    R, G, B = imgRGB[:,:,0], imgRGB[:,:,1], imgRGB[:,:,2]
    imgGray = (R+G+B)/3
    print(R,G,B)
    plt.imshow(imgGray,cmap='gray')
    plt.show()
    return imgGray
#gray1 = convertRGB_equal(LenaRGB)
T1=time.time()
def convertRGB_weighted1(imgRGB):
    #img = npimg.imread('lena.png')
    assert type(imgRGB) is np.ndarray

    R, G, B = imgRGB[:,:,0], imgRGB[:,:,1], imgRGB[:,:,2]
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    plt.imshow(imgGray,cmap='gray')
    plt.show()
    return imgGray
#gray2 = convertRGB_weighted1(LenaRGB)
T2=time.time()
print(T2-T1)
t1=time.time()
def convertRGB_weighted2(imgRGB):
    return np.dot(imgRGB, [0.299, 0.587, 0.114])

#img = npimg.imread('lena.png')
gray3 = convertRGB_weighted2(imgRGB)
#plt.imshow(gray3,cmap='gray')
#plt.show()
gray = convertRGB_weighted1(loadImage2Array('lena.png'))
#hist = plothistogram(gray)

t2=time.time()
print(t2-t1)
#plt.savefig('grey3.png')

def plothistogram(imgGray):
    imghistogram, bins = np.histogram(imgGray.flatten(),256)
    plt.subplot(121),plt.title('Histogram 1')#plt.xlim(0,255),plt.ylim(0,500)
    plt.plot(imghistogram)
    plt.subplot(122),plt.title('Histogram 2')#plt.xlim(0,255),plt.ylim(0,1500)
    plt.hist(imgGray.flatten(),bins=256)
    plt.show()
plothistogram(gray3)

#Conv1 = convertRGB_equal(arr)
#saveArray2GrayImg(Conv1,'Gray1')