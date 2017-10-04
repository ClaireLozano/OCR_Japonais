# Image Gaussien
import sys
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import imghdr
from scipy import ndimage
import scipy.misc
from scipy.signal import convolve as scipy_convolve
from scipy import signal as sg
from scipy import ndimage as ndi
from operator import itemgetter
import imageio
import matplotlib.image as mpimg
from  matplotlib.pyplot import *
import random

def applicationOfFilter(img, folder):

	# ==============================
	# ========== OPEN IMG ==========
	# ==============================

	# Ovrir image 
	img = openImage(folder, img)
	# Image noir et blanc
	imgBW = getNBImage(img)
	# Save image
	scipy.misc.imsave('otsu.png', imgBW)

	# PATH = "img/a.png"
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
	    # if h<40 or w<40:
	    #     continue
	    cv2.rectangle(img,(x-2,y-2),(x+w-12,y+h-12),(0,0,255),1) 

	cv2.imwrite('contoured_1.png', img)




	# image = 1
	# tab = []
	# width, height = imgBW.shape
	# for i in range(0, width)
	# 	for j in range(0, height)
	# 		if imgBW[i, j] = 0
	# 			xmin, ymin, xmax, ymax = getXYMinMax(width, height, 0, 0)
	# 			tab[]

	# plt.hist(img.ravel(),256,[0,256])
	# plt.show()
	# plt.imshow(im_med)
	# plt.show()

	# ==============================
	# ========== ANALYSE ===========
	# ==============================

	# Ovrir image 
	no = openImage(folder, "no.png")
	no2 = openImage(folder, "no2.png")
	a = openImage(folder, "a.png")
	shi = openImage(folder, "shi.png")
	tsu = openImage(folder, "tsu.png")
	nu = openImage(folder, "nu.png")

	# Image noir et blanc
	no = getNBImage(no)
	no2 = getNBImage(no2)
	a = getNBImage(a)
	shi = getNBImage(shi)
	tsu = getNBImage(tsu)
	nu = getNBImage(nu)

	# Pourcentage de noir et blanc 
	noHist = getHistGreyImage(no)
	aHist = getHistGreyImage(a)
	shiHist = getHistGreyImage(shi)
	tsuHist = getHistGreyImage(tsu)
	nuHist = getHistGreyImage(nu)

	# Create dictionnary
	percentTab = {}

	# Put values in a dictionnary
	percent = noHist[0]/(noHist[0]+noHist[255])*100
	percentTab["no"] = percent[0]
	percent = aHist[0]/(aHist[0]+aHist[255])*100
	percentTab["a"] = percent[0]
	percent = shiHist[0]/(shiHist[0]+shiHist[255])*100
	percentTab["shi"] = percent[0]
	percent = tsuHist[0]/(tsuHist[0]+tsuHist[255])*100
	percentTab["tsu"] = percent[0]
	percent = nuHist[0]/(nuHist[0]+nuHist[255])*100
	percentTab["nu"] = percent[0]

	print ""
	print "tableau de poucentage de noir : "
	print percentTab
	print "on peut dire que si c'est en dessous de 13, c'est shi ou tsu, sinon c'est a, nu ou no"
	print ""

	# SIFT
	# compare(no, no2)

	return 0

def getNBImage(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Otsu
	img = img.astype(np.uint8)
	ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	NBimage = np.invert(thresh)
	return NBimage

def compare(img1, img2):
	# copyMatrix = no
	# sift = cv2.xfeatures2d.SIFT_create() 
	# kp, des = sift.detectAndCompute(no,None)
	# img = cv2.drawKeypoints(no,kp, copyMatrix)
	# cv2.imwrite('sift_keypoints_no.jpg',img)

	# copyMatrix2 = no2
	# kp2, des2 = sift.detectAndCompute(no2,None)
	# img2 = cv2.drawKeypoints(no2,kp2, copyMatrix2)
	# cv2.imwrite('sift_keypoints_no2.jpg',img2)

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create() 

	# find the keypoints and descriptors with SIFT
	kp1,des1 = sift.detectAndCompute(img1,None)
	kp2,des2 = sift.detectAndCompute(img2,None)
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)
	matches = flann.knnMatch(des1,des2,k=2)
	# Need to draw only good matches, so create a mask
	matchesMask = [[0,0] for i in xrange(len(matches))]
	# ratio test as per Lowe's paper
	for i,(m,n) in enumerate(matches):
	    if m.distance < 0.7*n.distance:
	        matchesMask[i]=[1,0]
	draw_params = dict(matchColor = (0,255,0),
	                   singlePointColor = (255,0,0),
	                   matchesMask = matchesMask,
	                   flags = 0)
	img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
	cv2.imwrite('sift_keypoints_compare.jpg',img3)


# Return image opened
def openImage(folder,filename):
	return cv2.imread(os.path.join(folder,filename))

# Return the 3 histogram RGB
def getHistColorImage(img):
	histBlue = cv2.calcHist([img],[0],None,[16],[0,256])
	histGreen = cv2.calcHist([img],[1],None,[16],[0,256])
	histRed = cv2.calcHist([img],[2],None,[16],[0,256])
	return [histBlue, histGreen, histRed]

# Return the 3 histogram RGB
def getHistGreyImage(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	return hist


# Line : python tp_burie.py img.jpg
img = str(sys.argv[1])
folder = str(sys.argv[2])
applicationOfFilter(img, folder)
