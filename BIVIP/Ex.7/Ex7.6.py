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


#######################################################
'''
# Exercise 7.1
CAMPORT = 0
def takeImage( camObj ):
    for i in range(10):
        _,_ = camObj.read()
        print('.')
    ret, img = camObj.read()
    return ret, img
camObj = cv2.VideoCapture(CAMPORT)
if camObj.isOpened():
    res = (str(int(camObj.get(3))), str(int(camObj.get(4))))
    ret, img = takeImage(camObj)
    if ret is True:
        cv2.imwrite(PATH + 'testimage_' + res[0] + 'x' + res[1] + '.png',img)
isClosed = camObj.release

######################################################################
#Ex7.2
#CAMPORT = 0
RES = (640, 480)
RGB = BGR = 3
DEBUG = False

def showImage( img ):
    plt.close()
    plt.figure()
    plt.title('image'), plt.axis('off')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

words = {'key' : 'object', 'animal' : 'dog', 'hello' : 'Hallo'}
words['key']

info = {'start' : 'Starting panorama program\n',
        'init' : 'Init the Camera\n ',
        'initOK' : 'Camera init succeeded.\n ',
        'initFail' : 'Camera init failed, restarting...\n ',
        'frameCountFail' : 'Wrong frame number, using 4 images.\n',
        'settingsOK' : 'Settings completed, starting with the images.\n',
        'camError': 'camera error, restarting...\n',
        'closeWindow': 'To continue please close the image window.\n',
        'waitCam': 'Waiting for camera alignment...\n',
        'camEjectOK': 'Camera object successfully closed.\n',
        'camEjectFail': 'Closing camera object failed.\n'
                }

userInput = {'camport' : 'Choose the camera port (default 0): ',
            'res' : 'Set the resolution as (width, height) tuple: ',
            'frameCount' : 'How many single frames to capture (max.6)?',
            'startImgRec' : 'Press Enter to start the image recording.',
            'takeImg' :'Press [ENTER] to take a picture.',
            'contImgRec' : 'Press [ENTER] to continue.'}

def forceRun(port, resInput):
    ret = False
    #port = int(port)
    res = resInput.split(',')
    while ret is not True:
        camObj = cv2.VideoCapture(port)
        ret, img = takeImage(camObj)
    print(int(res[0]), int(res[1]))
    camObj.set(3, int(res[0]))
    camObj.set(4, int(res[1]))
    return camObj


def interface():
    frames = 0
    print(info['start'])
    print(info['init'])
    camPortInput = input('Enter the Camera ports:' )
    resInput=input('Enter the resolution:' )
    resInput=resInput[1:len(resInput) - 1]
    camObj = forceRun(camPortInput, resInput)
    if camObj.isOpened():
        print('the interface is open')
    else:
        print('the interface is closed')
        interface()
    res = resInput.split(',')
    frameNumber = input(' Enter the frame number:')
    frameNumber=int(frameNumber)
    if (1 < frameNumber) and (7 > frameNumber):
        print('Creating Panorama with ' + str(frameNumber) + ' frames')
    else:
        frameNumber = 4
        print('Failed')
    print('waiting for user confirmation')
    frames=np.zeros((int(res[1]), int(res[0]), BGR, frameNumber), dtype=np.uint8)
    if input(userInput['takeImg']) == "":
        pass
    while (frameNumber):
        if input(userInput['takeImg']) == "":
            ret, img=takeImage(camObj)
            if not ret:
                print('Error', info['camError'])
                interface()
            frames[:,:,:,-frameNumber] = img[:]
            print("")
            showImage(img)
            frameNumber = frameNumber - 1
            print('\nPicture saved, ' + str(frameNumber) + 'remaining...')
        else:
            print(info['waitCam'])
            while (input(userInput['contImgRec']) == ""):
                pass
    closed = camObj.release
    if closed:
        print('closed')
    else:
        print('CamError')
    return frames
'''
#Exercise 7.6
############################################################################

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





#######################################################################################
def stitch2Images(imgA, imgB, H):
    hA,wA = imgA.shape[:2]
    hB,wB = imgB.shape[:2]
    ptsA = np.float32([[0,0],[0,hA],[wA,hA],[wA,0]]).reshape(-1,1,2)
    ptsB = np.float32([[0,0],[0,hB],[wB,hB],[wB,0]]).reshape(-1,1,2)
    ptsBpt = cv2.perspectiveTransform(ptsB,H)
    pts = np.concatenate((ptsA, ptsBpt), axis=0)
    [mMin, nMin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [mMax, nMax] = np.int32(pts.max(axis=0).ravel() + 0.5)
    Ht = np.array([[1,0,-mMin],[0,1,-nMin],[0,0,1]])
    result = cv2.warpPerspective(imgB, Ht.dot(H), (mMax-mMin, nMax-nMin))
    result[-nMin:hA-nMin,-mMin:wA-mMin] = imgA
    return result

def createPanorama(imgA,imgB, features = 400, mask=None):
    sift = cv2.xfeatures2d.SIFT_create(FEATURES)
    kp1, des1 = sift.detectAndCompute(imgA, MASK)
    kp2, des2 = sift.detectAndCompute(imgB, MASK)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k=2)
    goodMatches = []
    for match1, match2 in matches:
        if match1.distance < 0.8 * match2.distance:
            goodMatches.append(match1)
    print("Good matches with Brute-Force found: %d" % (len(goodMatches)))

    featImgA = cv2.drawKeypoints(imgA, kp1, None, (0, 0, 255), 4)
    featImgB = cv2.drawKeypoints(imgB, kp2, None, (0, 0, 255), 4)
    cv2.imwrite('featImgA.png', featImgA)
    cv2.imwrite('featImgB.png', featImgB)
    dstPts = [kp1[i.queryIdx].pt for i in goodMatches]
    srcPts = [kp2[i.trainIdx].pt for i in goodMatches]
    H, mask = cv2.findHomography(np.float32(srcPts).reshape(-1, 1, 2),
                                     np.float32(dstPts).reshape(-1, 1, 2), cv2.RANSAC, ransacReprojThreshold=5.0)
    resultImg = stitch2Images(img1, img2, H)
    resultMatches = drawMatches(img1, kp1, img2, kp2, goodMatches)
    return featImgA, featImgB, resultMatches, resultImg


def processImgData( frames ):
    frameCount = frames.shape[3]
    stitchCount = frameCount - 1
    for i in range(frameCount):
        cv2.imwrite(PATH + 'img_' + str(i) + '.png', frames[:,:,:,i])
    for i in range(stitchCount):
        imgB = cv2.cvtColor(frames[:,:,:,i+1], cv2.COLOR_BGR2GRAY)
        if i == 0:
            imgA = cv2.cvtColor(frames[:,:,:,i], cv2.COLOR_BGR2GRAY)
        featImgA, featImgB, matches, imgA = createPanorama(imgA,imgB,400)
        cv2.imwrite(PATH + 'featImg_' + str(i) + '.png', featImgA)
        cv2.imwrite(PATH + 'featImg_' + str(i+1) + '.png', featImgB)
        cv2.imwrite(PATH + 'resultMatches_' + str(i) + str(i+1) + '.png',matches)
        if i is stitchCount - 1:
            cv2.imwrite(PATH + 'panorama.png', imgA)
        del(featImgA, featImgB, matches)
img1 = cv2.imread(PATH + 'A1.png', 0)
img2 = cv2.imread(PATH + 'A2.png', 0)
frames = 0
#processImgData(frames)
#createPanorama(img1,img2,400)
featImgA, featImgB, resultMatches, result12 = createPanorama(img1, img2, 400)
plt.imshow(result12,'gray'),plt.axis('off')
plt.show()

'''
#Exercise 7.7
import _thread as thread
DEBUG = False
def showFreeMemory():
    while( True ):
        with open('/Users/49174/meminfo','r') as mem:
            memlist = mem.readlines()
        if not mem.closed:
            mem.close()
        freeMem = memlist[2]
        print('free memory: ' + freeMem[13:-4].strip(' ') + ' kB')
        time.sleep(0.5)
if DEBUG:
    try:
        threadID = thread.start_new_thread(showFreeMemory, ())
        print('DEBUG MODE ON')
    except:
        print('thread creation failed!')
frames = interface()
processImgData(frames)
if DEBUG:
    try:
        threadID.exit()
    except:
        print('thread closing failed!')
'''
