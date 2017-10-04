# Image Gaussien
import sys
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import imghdr
from scipy import ndimage 
import scipy.misc
from astropy.io import fits
from astropy.utils.data import get_pkg_data_filename
from astropy.convolution import Gaussian2DKernel
from scipy.signal import convolve as scipy_convolve
from astropy.convolution import convolve
from scipy import signal as sg
from scipy import ndimage as ndi

def applicationOfFilter(img, folder):

	# ==============================
	# ========== OPEN IMG ==========
	# ==============================

	# Open initial image
	img = openImage(folder, img)
	plt.imshow(img)
	plt.show()
	# Get image in gray
	gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Otsu
	gray_image = gray_image.astype(np.uint8)
	ret, thresh = cv2.threshold(gray_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imgInvert = np.invert(thresh)

	# Save image
	scipy.misc.imsave('otsu.png', imgInvert)

	# plt.hist(img.ravel(),256,[0,256])
	# plt.show()
	# plt.imshow(im_med)
	# plt.show()

	# ==============================
	# ========= TRAITEMENT =========
	# ==============================


	return 0



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
