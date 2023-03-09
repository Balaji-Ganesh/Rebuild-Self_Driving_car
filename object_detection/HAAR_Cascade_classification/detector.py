# Importing OpenCV package
import cv2

# Reading the image
img = cv2.imread('stop_sign.png')

# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier(
	'./Stop sign detection/classifier/cascade.xml')

# Applying the face detection method on the grayscale image
sign_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
print(sign_rect)

# Iterating through rectangles of detected signs
for (x, y, w, h) in sign_rect:
	cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected signs', img)

cv2.waitKey(0)
