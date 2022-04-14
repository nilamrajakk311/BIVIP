# Exercise 3.1
#!users\49174\anaconda3\lib\site-packages (8.2.0)

import PIL
from PIL import Image as img
from PIL import ImageDraw as draw
import time
# Global variables
TRUE = 1
FALSE = 0
BLACK = 0
WHITE = 1
CHESSSIZE = [8,8]
FIELDSIZE = [5,5]
WIDTH = CHESSSIZE[0]*FIELDSIZE[0]
HEIGHT = CHESSSIZE[1]*FIELDSIZE[1]# TODO

# Exercise 3.2
def saveImg( pic, picName, picType ):
    assert type(picName) is type('')
    assert type(picType) is type('')
    try:
        pic.save(picName,format=picType)
        result = TRUE
    except:
        result = FALSE
    print(result)
    return result

def createBlankImg( picWidth, picHeight ):
    assert type(picWidth) is type(1)
    assert type(picHeight) is type(1)
    assert (picWidth > 0) and (picHeight > 0)
    pic = img.new(mode='RGB', size=(picWidth, picHeight))
    return pic

pic = createBlankImg(WIDTH,HEIGHT)
print(saveImg(pic,"myimage", "jpg"))

# Exercise 3.3
def createChessField1(pic):
    assert type(pic.size) is type(())
    assert (pic.size[0] == WIDTH) and (pic.size[1] == HEIGHT)
    bw = [BLACK, WHITE]
    chessField = pic.copy()
    for i in range(WIDTH):
        for j in range(HEIGHT):
            #chessField[i,j]=bw
            if(i//5+j//5)%2:
                chessField.putpixel((i,j),(255,255,255))
    return chessField
#pic = createBlankImg(40,40)
im = createChessField1(pic)

im.show()
saveImg(im,'chessfield1','pbm')

# Exercise 3.5
def openPic( filePath ):
    try:
        pic = img.open('C:/Users/49174/PycharmProjects/BIVIP/chessfield.pbm')
    except:
        result = FALSE
    return pic
pic = img.open('C:/Users/49174/PycharmProjects/BIVIP/chessfield.pbm')

def createBlackFrame(pic,frameWidth):
    assert type(frameWidth) is type(1)
    assert (frameWidth>= 0)
    w, h = pic.size[0], pic.size[1]
    shape = [(0, 0), (w, h)]
    picWithFrame = draw.Draw(pic)
    picWithFrame.rectangle(shape, outline="black", width=5)
    return picWithFrame
createBlackFrame(im,5)
def transposePic( pic ):
    rotatedPic = pic.transpose(img.FLIP_TOP_BOTTOM)
    return rotatedPic
pic = createBlankImg(40,40)
im1 = createChessField1(pic)
saveImg(im1, "chessfield2", "pbm")

transposePic(im1)

im.show(transposePic(im1))

# Exercise 3.6
def createBbox(position):
    x0,y0= (position[0],position[1])
    Area = (WIDTH-FIELDSIZE[0],HEIGHT-FIELDSIZE[1])
    assert (0 <= x0 <= Area[0]) and (0 <= y0 <= Area[1])

    #shape = [Area,(WIDTH,HEIGHT)]
    BlackBox = (x0,y0,FIELDSIZE[0]+x0,FIELDSIZE[1]+y0)

    #BlackBox.rectangle((shape, (WIDTH,HEIGHT)), fill="black")
    return BlackBox
def createChessField2(pic):
    chessField2 = pic.copy()
    for i in range (0,WIDTH,FIELDSIZE[0]*2):
        for j in range (0,HEIGHT,FIELDSIZE[1]*2):
           chessField2.paste(BLACK,createBbox((i,j)))
           chessField2.paste(BLACK,createBbox((i+FIELDSIZE[0],j+FIELDSIZE[1])))
    return chessField2

pic = createBlankImg(WIDTH,HEIGHT)
im = createChessField2(pic)

im.show()
saveImg(im,'chessfield3','pbm')
#3.7
import timeit
def createChessField3( pic ):
    assert type(pic.size) is type(())
    assert (pic.size[0] == WIDTH) and (pic.size[1] == HEIGHT)
    bw = [BLACK, WHITE]
    chessField = pic.copy()
    chessdraw = draw.Draw(chessField)
    t1 = time.time()
    for i in range(WIDTH):
        for j in range(HEIGHT):
            chessdraw.point((i, j), bw[(i//5+j//5+2) % 2])
    t2 = time.time()
    print(t2-t1)
    return chessField
pic = createBlankImg(WIDTH,HEIGHT)
im3= createChessField3(pic)
im3.show()
saveImg(im3, 'test3', 'pbm')
# Exercise 3.8
# Code example of a runtime measurement
def checkTime( timeInSec ):
    start = time.time()
    time.sleep(timeInSec)
    stop = time.time()
    return stop-start

pic = createBlankImg(WIDTH,HEIGHT)
im =createChessField1(pic)
#checkTime(timeInSec)