import os 
import cv2
import numpy as np
from matplotlib import pyplot as plt
import imghdr
from operator import itemgetter
import imageio
import matplotlib.image as mpimg
from  matplotlib.pyplot import *
import random
from scipy import ndimage, misc
from skimage import measure
from skimage import segmentation
from skimage import morphology


PATH = "otsu.png"

img = cv2.imread(PATH)

# Transformation en niveau de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Seuillage
_,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

# Dillatation
dilated = cv2.dilate(thresh,kernel,iterations = 13)

# Detection des contours
val1 = cv2.RETR_EXTERNAL
val2 = cv2.CHAIN_APPROX_NONE

_, contours, _= cv2.findContours(dilated, val1, val2)

# Pour chaque contour on les redessine sur l'image original
for contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)

    # On supprime les zones trop petites, pour eviter le bruit
    if h<40 or w<40:
        continue

    if h > 200 and w > 200:
        cv2.putText(img,"Image", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 8, True)

    cv2.rectangle(img,(x-2,y-2),(x+w,y+h),(0,0,255),2)

cv2.imwrite('contoured_1.png', img)

