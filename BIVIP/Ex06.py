import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import filters
import scipy.ndimage as ndim

# Exercise 6.1
path = '/Users/49174/PycharmProjects/BIVIP/'
def createTestImg():
    img = np.zeros((800,800),dtype = np.uint8)
    img[:] = 255
    img[:400,:400] = 0
    return img
def addNoise(img, dev):
    nImg = img + np.random.normal(loc=0,scale=dev,size=img.shape)
    return nImg
def computeDerivatives(img, sigma=3):
    Fm = np.zeros(img.shape)
    Fn = np.zeros(img.shape)
    ndim.gaussian_filter(img, (sigma, sigma), (0, 1), Fm)
    ndim.gaussian_filter(img, (sigma, sigma), (1, 0), Fn)
    return Fm, Fn
def windowImg(imgWindow, img, Fm, Fn):
    frameSize = 20
    red = (255,0,0)
    m0, n0, m1, n1 = imgWindow[0],imgWindow[1],imgWindow[2],imgWindow[3]
    imgWithFrame = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    imgWithFrame[m0:m1,n0:n0+frameSize] = red
    imgWithFrame[m0:m1,n1-frameSize:n1] = red
    imgWithFrame[m0:m0+frameSize,n0:n1] = red
    imgWithFrame[m1-frameSize:m1,n0:n1] = red
    wFm = Fm[m0:m1, n0:n1]
    wFn = Fn[m0:m1,n0:n1]
    return wFm, wFn, imgWithFrame
img = createTestImg()
nImg = addNoise(img, 3)
Fm, Fn = computeDerivatives(img)
FmN,FnN = computeDerivatives(nImg)
flat, edge, corner = (600,200,700,300), (350,100,450,200), (350,350,450,450)
wFm1, wFn1, img1 = windowImg(flat, img, Fm, Fn)
wFm2, wFn2, img2 = windowImg(edge, img, Fm, Fn)
wFm3, wFn3, img3 = windowImg(corner, img, Fm, Fn)
wFm4, wFn4, img4 = windowImg(flat, np.uint8(nImg), FmN, FnN)
wFm5, wFn5, img5 = windowImg(edge, np.uint8(nImg), FmN, FnN)
wFm6, wFn6, img6 = windowImg(corner, np.uint8(nImg), FmN, FnN)

# Exercise 6.1 continued
plt.figure(1)
plt.subplot(4,3,1), plt.title('noiseless'), plt.axis('off')
plt.gray(), plt.imshow(img)

plt.subplot(4,3,4),plt.title('Flat'),plt.axis('off')

plt.gray(), plt.imshow(img1)
plt.subplot(4,3,5),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm1)
plt.subplot(4,3,6),plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn1)

plt.subplot(4,3,7),plt.title('edge'),plt.axis('off')
plt.gray(), plt.imshow(img2)
plt.subplot(4,3,8),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm2)
plt.subplot(4,3,9),plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn2)

plt.subplot(4,3,10),plt.title('corner'),plt.axis('off')
plt.gray(), plt.imshow(img3)
plt.subplot(4,3,11),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm3)
plt.subplot(4,3,12), plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn3)
plt.show()

plt.figure(2)
plt.subplot(4,3,1), plt.title('noisy'), plt.axis('off')
plt.gray(), plt.imshow(nImg)
plt.subplot(4,3,4),plt.title('noisy Flat'),plt.axis('off')
plt.gray(), plt.imshow(img4)
plt.subplot(4,3,5),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm4)
plt.subplot(4,3,6),plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn4)

plt.subplot(4,3,7),plt.title('noisy edge'),plt.axis('off')
plt.gray(), plt.imshow(img5)
plt.subplot(4,3,8),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm5)
plt.subplot(4,3,9),plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn5)

plt.subplot(4,3,10),plt.title('noisy corner'),plt.axis('off')
plt.gray(), plt.imshow(img6)
plt.subplot(4,3,11),plt.title('Fm'),plt.axis('off')
plt.gray(), plt.imshow(wFm6)
plt.subplot(4,3,12), plt.title('Fn'),plt.axis('off')
plt.gray(), plt.imshow(wFn6)
plt.show()

# Exercise 6.2
plt.figure(3)
plt.subplot(321), plt.title('flat')
plt.ylabel('Fn')
plt.axis([-10, 10, -10, 10])
plt.grid(True)
plt.plot(wFm1[:], wFn1[:], 'rx')
plt.subplot(322), plt.title('noisy flat')
plt.axis([-10, 10, -10, 10])
plt.grid(True)
plt.plot(wFm4[:], wFn4[:], 'rx')
plt.subplot(323), plt.title('edge')
plt.ylabel('Fn')
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.plot(wFm2[:], wFn2[:], 'rx')

plt.subplot(324), plt.title('noisy edge')
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.plot(wFm5[:], wFn5[:], 'rx')

plt.subplot(325), plt.title('corner')
plt.ylabel('Fn')
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.plot(wFm3[:], wFn3[:], 'rx')

plt.subplot(326), plt.title('noisy corner')
plt.axis([-40, 40, -40, 40])
plt.grid(True)
plt.plot(wFm6[:], wFn6[:], 'rx')
plt.show()


#Exercise 6.3
def computeHarrisCorners(Fm, Fn, k=0.04, sigma=3):
    Mmm = filters.gaussian_filter(Fm**2,sigma)
    Mmn = filters.gaussian_filter(Fm*Fn,sigma)
    #Mnm = filters.gaussian_filter(Fn*Fm,sigma)
    Mnm = Mmn

    Mnn = filters.gaussian_filter(Fn**2,sigma)
    detM = Mmm*Mnn-Mmn*Mmn
    traM = Mmm+Mnn
    R = detM-k*traM**2
    return Mmm, Mmn, Mnm, Mnn, detM, traM, R
Mmm1, Mmn1, Mnm1, Mnn1, detM1, traM1, R1 = computeHarrisCorners(Fm,Fn)
Mmm2, Mmn2, Mnm2, Mnn2, detM2, traM2, R2 = computeHarrisCorners(FmN,FnN)

plt.figure(4)
plt.subplot(221), plt.title('Image')
plt.imshow(img), plt.axis('off')
plt.subplot(222), plt.title('Corner response R')
plt.imshow(R1),plt.axis('off')
plt.subplot(223), plt.title('det (M)')
plt.imshow(detM1),plt.axis('off')
plt.subplot(224), plt.title('trace(M)')
plt.imshow(traM1),plt.axis('off')
plt.show()

plt.figure(5)
plt.subplot(221), plt.title('noisy Image')
plt.imshow(nImg), plt.axis('off')
plt.subplot(222), plt.title('Corner response R')
plt.imshow(R2),plt.axis('off')
plt.subplot(223), plt.title('det (M)')
plt.imshow(detM2),plt.axis('off')
plt.subplot(224), plt.title('trace(M)')
plt.imshow(traM2),plt.axis('off')
plt.show()

fig = plt.figure(6)
ax = fig.gca(projection='3d')
x = np.arange(R1.shape[1])
y = np.arange(R1.shape[0])
X, Y = np.meshgrid(x, y)
Z = R1
ax.plot_surface(X, Y, Z, rstride=20, cstride=20, alpha = 0.3)
cset = ax.contour(X, Y, Z, zdir='z')
cset = ax.contour(X, Y, Z, zdir='x')
cset = ax.contour(X, Y, Z, zdir='y')
ax.set_xlabel('m')
ax.set_ylabel('n')
ax.set_zlabel('R(m,n)')
plt.show()
fig = plt.figure(7)
ax = fig.gca(projection='3d')
x = np.arange(R2.shape[1])
y = np.arange(R2.shape[0])
X, Y = np.meshgrid(x, y)
Z = R2
ax.plot_surface(X, Y, Z, rstride=20, cstride=20, alpha = 0.3)
cset = ax.contour(X, Y, Z, zdir='z')
cset = ax.contour(X, Y, Z, zdir='x')
cset = ax.contour(X, Y, Z, zdir='y')
ax.set_xlabel('m')
ax.set_ylabel('n')
ax.set_zlabel('R(m,n)')
plt.show()


def detectCorners( R, threshold=0.3):
    cornersCandidates = np.uint8((R>R.max()*threshold))
    cornerPos = np.array(cornersCandidates.nonzero()).T
    return cornerPos
cornerPos1 = detectCorners(R1)
cornerPos2 = detectCorners(R2)

plt.figure(8)
plt.subplot(221), plt.title('corners in image'), plt.axis('off')
plt.gray(), plt.imshow(img), plt.plot(cornerPos1[:,1],cornerPos1[:,0],'ro')
plt.subplot(223)
plt.text(0.2, 0.8, str(cornerPos1.shape[0]) + ' corners found', fontsize=14)
plt.axis('off')
plt.subplot(222),plt.title('corners in noisy image'), plt.axis('off')
plt.gray(), plt.imshow(nImg), plt.plot(cornerPos2[:,1],cornerPos2[:,0],'ro')

plt.subplot(224)
plt.text(0.2, 0.8, str(cornerPos2.shape[0]) + ' corners found', fontsize=14)
plt.axis('off')

plt.show()
#Exercise 6.4

def rotateImg(img, deg=45, scale=1.0):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),deg, scale)
    return cv2.warpAffine(img, M, (w,h) )
IMG1 = rotateImg(img, deg=45, scale=1.0)
IMG2 = rotateImg(nImg, deg=45, scale=1.0)
plt.figure(9)
plt.subplot(121), plt.imshow(IMG1),plt.axis('off'),plt.title('noiseless img')
plt.subplot(122), plt.imshow(IMG2), plt.axis('off'), plt.title('noisy img')
plt.show()


#Exercise 6.5
def importAndConvertImg( imgPath, imgSize=800 ):
    img = cv2.imread(imgPath)
    h, w = img.shape[:2]
    if h != imgSize:
        scaleFactor = imgSize/h
        img = cv2.resize(img, (int(w*scaleFactor), (int(h*scaleFactor))), interpolation=cv2.INTER_LINEAR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
imgPath ='/Users/49174/PycharmProjects/BIVIP/Chessboard.png'
convertChess = importAndConvertImg(imgPath,800)
FM,FN = computeDerivatives(np.uint8(convertChess))
Mmm, Mmn, Mnm, Mnn, detM, traM, R = computeHarrisCorners(FM,FN,k=0.04,sigma=3)
cornerPos = detectCorners(R,0.05)

plt.figure(10)
plt.subplot(321),plt.axis('off')
plt.imshow(convertChess),plt.title('Image')
plt.subplot(322),plt.axis('off')
plt.imshow(R),plt.title('corner response R')
plt.subplot(323),plt.axis('off')
plt.imshow(detM),plt.title('det(M)')
plt.subplot(324),plt.axis('off')
plt.imshow(traM), plt.title('trace(M)')
plt.subplot(325),plt.axis('off')
plt.gray(), plt.imshow(convertChess)
plt.plot(cornerPos[:,0],cornerPos[:,1],'ro'), plt.title('corners in image')  #############
plt.subplot(326)
plt.text(0.2, 0.8, str(len(cornerPos)) + ' corners found', fontsize=14)
plt.axis('off')
plt.show()
