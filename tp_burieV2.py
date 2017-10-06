from __future__ import division
import sys
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
import scipy.misc
from scipy.signal import convolve as scipy_convolve
from  matplotlib.pyplot import *

def findCaractere(img, folder):
	# Get BDD 
	BDDNonComplexe, BDDComplexe = makeBDD()

	# Nettoyer dossier analyse
	EraseFile(folder + "/analyse")

	# Ovrir image / mettre en noir et blanc / save
	img = openImage(folder, img)
	imgBW = getNBImage(img)
	scipy.misc.imsave(folder + '/otsu.png', imgBW)

	# Dillatation
	imgBW = np.invert(imgBW)
	kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))
	dilated = cv2.dilate(imgBW,kernel,iterations = 13)

	# Detection des contours
	val1 = cv2.RETR_EXTERNAL
	val2 = cv2.CHAIN_APPROX_NONE
	_, contours, _= cv2.findContours(dilated, val1, val2)

	# index 
	i = 0
	# Pour chaque caracteres detecte, on crop l'image avant de l'analyser et la comparer
	for contour in contours:
		[x,y,w,h] = cv2.boundingRect(contour)
		# On supprime les zones trop grandes ou trop petite, pour eviter le bruit
		if h<30 or w<30:
			continue

		# Dessiner les rectangles
	    # cv2.rectangle(img,(x-2,y-2),(x+w-12,y+h-12),(0,0,255),1)

	    # Crop image
		imgCrop = img[y-2:y+h-12, x-2:x+w-12]


		res = analyseImage(imgCrop, "caractereDetecte_" + str(i))
		scipy.misc.imsave(folder + "/analyse/caractereDetecte_" + str(i) + ".png", imgCrop)

		# Comparer le pourcentage de pixel noir pour les classifier
		# Ces valeurs sont choisit arbitrairement
		if( res[1] < 15):
			# Si pas beaucoup de pixel, alors le caractere et soit un "tsu" soit un "shi"
			compare(res, BDDNonComplexe)

		if( res[1] > 15):
			# Sinon c'est un "a", "no" ou "nu"
			compare(res, BDDComplexe)
		
		# Solution que l'on avait utilise au debut : comparaison de l'element avec tout les element de la base de donnees
		# compare(res, BDD)
			
		i = i+1


# Analyse d'une image
def analyseImage(imgCrop, name):
	# Get pourcentage noir et blanc
	imgNB = getNBImage(imgCrop)
	hist = getHistGreyImage(imgNB)
	percent = hist[0]/(hist[0]+hist[255])*100

	# Initiate SURF detector
	# sift = cv2.xfeatures2d.SIFT_create() 
	surf = cv2.xfeatures2d.SURF_create()
	# find the keypoints and descriptors with SIFT
	# kp1,des1 = sift.detectAndCompute(imgCrop,None)
	kp1,des1 = surf.detectAndCompute(imgCrop,None)

	# Ajout du nom de l'image, pourcentage de pixel noir, point d'interet, descripteur, image
	distance = []
	distance.append(name)
	distance.append(percent[0])
	distance.append(kp1)
	distance.append(des1)
	distance.append(imgCrop)

	return distance


# Creation de la base de donnees 
def makeBDD():
	# Ovrir les images 
	no = openImage(folder, "no.png")
	a = openImage(folder, "a.png")
	shi = openImage(folder, "shi.png")
	tsu = openImage(folder, "tsu.png")
	nu = openImage(folder, "nu.png")

	# Creation du dictionnaire
	caract = {}
	caract["no"] = no
	caract["a"] = a
	caract["tsu"] = tsu
	caract["shi"] = shi
	caract["nu"] = nu

	# Creation des array qui representeront les bases de donnees
	BDDNonComplexe = []
	BDDComplexe = []

	for key, value in caract.iteritems():
		analyse = analyseImage(value, key)
		if (analyse[1] < 15):
			BDDNonComplexe.append(analyseImage(value, key))
		else:
			BDDComplexe.append(analyseImage(value, key))

	return BDDNonComplexe, BDDComplexe


# Analyse l'image avec toutes les images de la BDD
def compare(res, BDD):
	# FLANN parameters
	FLANN_INDEX_KDTREE = 1
	index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
	search_params = dict(checks=50)   # or pass empty dictionary
	flann = cv2.FlannBasedMatcher(index_params,search_params)

	# Recuperation des caracteristiques de l'image a analyser
	des1 = res[3]
	kp1 = res[2]
	img1 = res[4]
	
	# Initialisation des nombres de match pour chaque caratere de la bdd
	numbrMatche_a = 0
	numbrMatche_no = 0
	numbrMatche_shi = 0
	numbrMatche_nu = 0
	numbrMatche_tsu = 0

	numberMatches = 0
	matchesDictonary = {}
	tabKey = []     
	tabValue = []
	
	for element in BDD:
		# Recuperation des caracteristiques des images comprise dans la base de donnees
		des2 = element[3]
		kp2 = element[2]
		img2 = element[4]

		nearesstMatche = ''

		# Analyse des distances grace a la methode flann
		matches = flann.knnMatch(des1,des2,k=2)
		# Need to draw only good matches, so create a mask
		matchesMask = [[0,0] for i in xrange(len(matches))]
		# ratio test as per Lowe's paper
		for i,(m,n) in enumerate(matches):
			matched = 'false'
			numberMatches += 1
			if m.distance < 0.7*n.distance:
			    matchesMask[i]=[1,0]
			    matched = 'true'

			if (matched == 'true' and element[0] == 'a'):
				numbrMatche_a +=1
				matchesDictonary['a'] = numbrMatche_a
			if (matched == 'true' and element[0] == 'no'):
				numbrMatche_no +=1
				matchesDictonary['no'] = numbrMatche_no
			if (matched == 'true' and element[0] == 'shi'):
				numbrMatche_shi +=1
				matchesDictonary['shi'] = numbrMatche_shi
			if (matched == 'true' and element[0] == 'nu'):
				numbrMatche_nu +=1
				matchesDictonary['nu'] = numbrMatche_nu
			if (matched == 'true' and element[0] == 'tsu'):
				numbrMatche_tsu +=1
				matchesDictonary['tsu'] = numbrMatche_tsu
		
		draw_params = dict(matchColor = (0,255,0),
		                   singlePointColor = (255,0,0),
		                   matchesMask = matchesMask,
		                   flags = 0)
		img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
		cv2.imwrite(folder + "/analyse/sift_keypoints_compare" + element[0] + res[0] + ".jpg",img3)
	
	for key, value in sorted(matchesDictonary.iteritems(), key=lambda (k,v): (v,k)):
		tabKey.append(key)
		tabValue.append(value)
	
	recognizedElement = list(reversed(tabKey))	
	recognizationValue = list(reversed(tabValue))	

	if recognizationValue and recognizedElement:
		# Si il n'y a pas de match entre les points, ne pas le prendre en compte
		if (numberMatches == 0):
			return 0
		else:
			percentReconize(recognizedElement[0], recognizationValue[0], numberMatches, res[0])
	print ""


# Afficher le resultat de la comparaison de caractere
def percentReconize(recognizedElement, recognizationValue, numberMatches, currentImge):
	recognizationPercent = recognizationValue / numberMatches

	print ""
	# currentImage est res[0], nom de l'imgae a tester
	if(recognizationPercent > 0.05):
		recognizationPercent =  "{:.1%}".format(recognizationPercent)
		print "Un caractere a ete detecte et reconnu :"
		print "    l'image", currentImge, "a ete reconnu comme etant un '", recognizedElement, "' avec", recognizationValue, "sur", numberMatches, "match et donc", recognizationPercent, "de reconnaissance (l'image a ete enregistrer dans le dossier img/analyse)." 
	else:
		recognizationPercent =  "{:.1%}".format(recognizationPercent)
		print "Un caractere a ete detecte mais pas reconnu :"
		print "    l'image", currentImge, "n'a pas ete reconnu mais le caractere '", recognizedElement, "' semble etre le plus proche avec", recognizationValue, "sur", numberMatches, "match et donc", recognizationPercent, "de reconnaissance (l'image a ete enregistrer dans le dossier img/analyse)." 


# Retourne une image binariser - methode utilisee : Otsu
def getNBImage(img):
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = img.astype(np.uint8)
	ret, thresh = cv2.threshold(img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
	NBimage = np.invert(thresh)
	return NBimage


# Retourne image ouverte
def openImage(folder,filename):
	return cv2.imread(os.path.join(folder,filename))


# Retourne histogram de niveau de gris 
def getHistGreyImage(img):
	hist = cv2.calcHist([img],[0],None,[256],[0,256])
	return hist


# Supprime le contenue du dossier analyse
def EraseFile(folder):
	files=os.listdir(folder)
	for i in range(0,len(files)):
		os.remove(folder+'/'+files[i])


# Line : python tp_burieV2.py img.jpg img

# Image qui sera analysee
img = str(sys.argv[1])
# Localisation du dossier
folder = str(sys.argv[2])
findCaractere(img, folder)
