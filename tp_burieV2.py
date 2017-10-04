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

# ==============================
# ======= GET CARACTERES =======
# ==============================
def applicationOfFilter(img, folder):
	# Get BDD 
	BDD = makeBDD()

	# Ovrir image / mettre en noir et blanc / save
	img = openImage(folder, img)
	imgBW = getNBImage(img)
	scipy.misc.imsave('otsu.png', imgBW)

	# ouvrir image noir et blanc / transformer en gris / binaire 
	PATH = "otsu.png"
	img = cv2.imread(PATH)
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	_,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

	# Dillatation
	dilated = cv2.dilate(thresh,kernel,iterations = 13)

	# Detection des contours
	val1 = cv2.RETR_EXTERNAL
	val2 = cv2.CHAIN_APPROX_NONE

	_, contours, _= cv2.findContours(dilated, val1, val2)

	i = 0
	# Pour chaque contour on les redessine sur l'image original
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		# On supprime les zones trop grandes, pour eviter le bruit
		if h>300 or w>300:
			continue

	    # cv2.rectangle(img,(x-2,y-2),(x+w-12,y+h-12),(0,0,255),1)

	    # Crop image
		imgCrop = img[y-2:y+h-12, x-2:x+w-12]

		# Analyse image 
		res = analyseImage(imgCrop, "test" + str(i))
		# scipy.misc.imsave("test" + str(i) + ".png", imgCrop)

		# Compare
		compare(res, BDD)
		i = i+1

		# plt.hist(img.ravel(),256,[0,256])
		# plt.show()
		# plt.imshow(im_med)
		# plt.show()


# ==============================
# ========== ANALYSE ===========
# ==============================
def analyseImage(imgCrop, name):
	# Get pourcentage noir et blanc
	imgNB = getNBImage(imgCrop)
	hist = getHistGreyImage(imgNB)
	percent = hist[0]/(hist[0]+hist[255])*100

	# Initiate SIFT detector
	sift = cv2.xfeatures2d.SIFT_create() 

	# find the keypoints and descriptors with SIFT
	kp1,des1 = sift.detectAndCompute(imgCrop,None)

	distance = []
	distance.append(name)
	distance.append(percent[0])
	distance.append(kp1)
	distance.append(des1)
	distance.append(imgCrop)

	return distance


# ==============================
# ========== MAKE BDD ==========
# ==============================
def makeBDD():
	# Ovrir image 
	no = openImage(folder, "no.png")
	a = openImage(folder, "a.png")
	shi = openImage(folder, "shi.png")
	tsu = openImage(folder, "tsu.png")
	nu = openImage(folder, "nu.png")

	caract = {}
	caract["no"] = no
	caract["a"] = a
	caract["tsu"] = tsu
	caract["shi"] = shi
	caract["nu"] = nu

	BDD = []
	for key, value in caract.iteritems():
		BDD.append(analyseImage(value, key))

	# Return BDD
	return BDD

# ==============================
# ======= GET NB IMAGE =========
# ==============================
def getNBImage(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Otsu
	img = img.astype(np.uint8)
	ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	NBimage = np.invert(thresh)
	return NBimage

# ==============================
# =========== ANALYSE ==========
# ==============================
# Analyse l'image avec toutes les images de la BDD
def compare(res, BDD):
	# copyMatrix = no
	# sift = cv2.xfeatures2d.SIFT_create() 
	# kp, des = sift.detectAndCompute(no,None)
	# img = cv2.drawKeypoints(no,kp, copyMatrix)
	# cv2.imwrite('sift_keypoints_no.jpg',img)

	# copyMatrix2 = no2
	# kp2, des2 = sift.detectAndCompute(no2,None)
	# img2 = cv2.drawKeypoints(no2,kp2, copyMatrix2)
	# cv2.imwrite('sift_keypoints_no2.jpg',img2)

	# # Initiate SIFT detector
	# sift = cv2.xfeatures2d.SIFT_create() 

	# # find the keypoints and descriptors with SIFT
	# kp1,des1 = sift.detectAndCompute(img1,None)
	# kp2,des2 = sift.detectAndCompute(img2,None)


	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)

	des1 = res[3]
	kp1 = res[2]
	img1 = res[4]
	
	for element in BDD:
		des2 = element[3]
		kp2 = element[2]
		img2 = element[4]

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
		cv2.imwrite("sift_keypoints_compare" + element[0] + res[0] + ".jpg",img3)


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
