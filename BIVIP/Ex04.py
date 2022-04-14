# Code for the plots of Figure 5
import numpy as np
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import matplotlib.image as npimg
from PIL import Image
import cv2


#4.2
LenaRGB = np.array(Image.open('Lena.png'), dtype=np.uint8)
strImgSize = str(LenaRGB.size)
strImgHeight = str(LenaRGB.shape[0])
strImgWidth = str(LenaRGB.shape[1])
strImgChannels = str(LenaRGB.shape[2])
strImgDatatype = str(LenaRGB.dtype)
LenaRedCh = LenaRGB.copy()
LenaGreenCh = LenaRGB.copy()
LenaBlueCh = LenaRGB.copy()

LenaRedCh[:,:,1:3] = 0
LenaGreenCh[:,:,0] = 0
LenaGreenCh[:,:,2] = 0
LenaBlueCh[:,:,:2] = 0
plt.subplot(231),plt.imshow(LenaRGB),plt.title('Lena RGB')
plt.axis('off')
plt.subplot(232)
plt.text(0,.8,'Image Size:'),plt.text(0,.6,'Image Width:')
plt.text(0,.5,'Image Height:'),plt.text(0,.3,'Image Channels:')
plt.text(0,.2,'Image Datatype:'),plt.axis('off')
plt.subplot(233)
plt.text(0,.8,strImgSize),plt.text(0,.6,strImgWidth)
plt.text(0,.5,strImgHeight),plt.text(0,.3,strImgChannels)
plt.text(0,.2,strImgDatatype),plt.axis('off')
plt.subplot(234),plt.imshow(LenaRedCh)
plt.title('Red Channel'),plt.axis('off')
plt.subplot(235),plt.imshow(LenaGreenCh)
plt.title('Green Channel'),plt.axis('off')
plt.subplot(236),plt.imshow(LenaBlueCh)
plt.title('Blue Channel'),plt.axis('off')
plt.show()

