import cv2

PATH = "otsu.png"

img = cv2.imread(PATH)

# Transformation en niveau de gris
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Seuillage
_,thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# Création du partern de reconnaissance
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(2,2))

# Dillatation
dilated = cv2.dilate(thresh,kernel,iterations = 13)

# Detection des contours
contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

# Pour chaque contour on les redessine sur l'image original
for contour in contours:
    [x,y,w,h] = cv2.boundingRect(contour)

    # On supprime les zones trop petites, pour eviter le bruit
    if h<40 or w<40:
        continue

    # Dectection des images grâce a leur taille
    if h > 200 and w > 200:
        cv2.putText(img,"Image", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 2, 255, 8, True)

    # Dessin du rectangle de contour
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

# Enregistrement de l'image
cv2.imwrite('contoured.jpg', img)