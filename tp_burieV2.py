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

def applicationOfFilter(img):

	# ==============================
	# =========== PART 1 ===========
	# ==============================

	# Open initial image
	imageName1 = "Image1_gaussien_10.png"
	img = openImage(folder, imageName1)
	
	# Get image in gray
	gray_image1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

	# Crop image
	crop_img1 = gray_image1[970:1200, 870:1900] 

	scipy.misc.imsave('crop1.png', crop_img1)

	# Use median filter on the image
	im_med = ndimage.median_filter(crop_img4, 3)
	
	# Save Image
	scipy.misc.imsave('med1.png', im_med)
	
	# Convolution
	k = np.array([[1,1,1],[1,1,1],[1,1,1]])
	k = k / 9.0
	kernel = Gaussian2DKernel(stddev=1)

	im_convolve1 = scipy_convolve(crop_img1, k, mode='same')

	# Otsu
	im_convolve1 = im_convolve1.astype(np.uint8)
	ret, thresh1 = cv2.threshold(im_convolve1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imgInvert1 = np.invert(thresh1)

	kernel = np.ones((2,2), np.uint8)
	# img_erosion = cv2.erode(im_med3, kernel, iterations=1)
	img_dilate = cv2.dilate(im_med3, kernel, iterations=1)

	# Save image
	scipy.misc.imsave('otsu1.png', imgInvert1)

	# plt.hist(img.ravel(),256,[0,256])
	# plt.show()
	# plt.imshow(im_med)
	# plt.show()

	# ==============================
	# =========== PART 2 ===========
	# ==============================

	# Get open initial image
	imageName1 = "image2c.png"
	image3 = openImage(folder2, imageName3)

	# Get image in gray
	grayImage1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	grayImage1 = grayImage1.astype(np.uint8)
	
	ret, thresh1 = cv2.threshold(grayImage1,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	imgInvert1 = np.invert(thresh1)

	# img_erosion = cv2.erode(imgInvert1, kernel, iterations=2)
	img_dilate = cv2.dilate(imgInvert1, kernel, iterations=1)

	# # Save image
	scipy.misc.imsave('otsu1Color.png', imgInvert1)

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
CBIR_BDD = str(sys.argv[1])
applicationOfFilter(CBIR_BDD)
