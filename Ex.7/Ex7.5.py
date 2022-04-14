import numpy as np
import cv2, time
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from scipy.ndimage import filters
import scipy.ndimage as ndim


PATH = '/Users/49174/PycharmProjects/BIVIP/'
FEATURES = 400
MASK = None
sift = cv2.xfeatures2d.SIFT_create(FEATURES)


#Exercise 7.5
def drawMatches(img1, kp1, img2, kp2, matches):
    h1,w1 = img1.shape[:2]
    h2,w2 = img2.shape[:2]
    matchesImg = np.zeros((max([h1,h2]), w1+w2, 3), dtype='uint8')
    matchesImg[:h1,:w1] = np.dstack([img1, img1, img1])
    matchesImg[:h2,w1:] = np.dstack([img2, img2, img2])
    for i in matches:
        img1Idx = i.queryIdx
        img2Idx = i.trainIdx
        (m1,n1) = kp1[img1Idx].pt
        (m2,n2) = kp2[img2Idx].pt
        cv2.circle(matchesImg, (int(m1),int(n1)), 5, (0, 255, 0), 1)
        cv2.circle(matchesImg, (int(m2)+w1,int(n2)), 5, (0, 255, 0), 1)
        cv2.line(matchesImg,(int(m1),int(n1)),(int(m2)+w1,int(n2)),(0,255, 0), 1)
    return matchesImg
img1 = cv2.imread(PATH +'A1.png', 0)
img2 = cv2.imread(PATH + 'A2.png', 0)
kp1, des1 = sift.detectAndCompute(img1, MASK)
kp2, des2 = sift.detectAndCompute(img2, MASK)



bf = cv2.BFMatcher(cv2.NORM_L2)
matches = bf.knnMatch(des1,des2,k=2)
goodMatches = []
for match1,match2 in matches:
    if match1.distance<0.8*match1.distance:
        goodMatches.append(match1)
print("Good matches with Brute-Force found: %d" % (len(goodMatches)))
featImgA = cv2.drawKeypoints(img1, kp1, None, (0, 0, 255), 4)
featImgB = cv2.drawKeypoints(img2, kp2, None, (0, 0, 255), 4)
cv2.imwrite('featImgA.png', featImgA)
cv2.imwrite('featImgB.png', featImgB)
resultMatches = drawMatches(img1, kp1, img2, kp2, goodMatches)
#plt.imshow(resultMatches)
#plt.show()


