# ###################################################################################
# Exercise 9.6

# library imports
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

def pasteImage(alpha_grabcut, background_frame, w, h):

    # read images
    ag_img = cv2.imread('alpha_johhny_grabcut.png',-1)
    b_img = cv2.imread('beach.jpg')
    # set x & y coordinates
    x = 0
    y = 0
    # verification of forwarded data
    if ag_img is not None:
        print("ag_img: " + str(ag_img.shape))
    else:
        print("no ag_img")
    if b_img is not None:
        print("b_img: " + str(b_img.shape))
    else:
        print("no b_img")

    # scale alpha channel, create FG mask and determine BG mask
    alpha_ag = ag_img[:, :,  3] / 255.0
    alpha_b = 1.0 - alpha_ag

    # actual image pasting
    for c in range(0, 3):
        b_img[y:y + h, x:x + w, c] = (alpha_ag * ag_img[:, :, c] + alpha_b * b_img[y:y + h, x:x + w, c])

    cv2.imwrite("paste_img_johnny.jpg",b_img)
    plt.imshow(b_img), plt.show()

t1 = time.time()
pasteImage('alpha_grabcut.png','beach.jpg',700,466)
t2 = time.time()
t = t2 -t1
print(t)
