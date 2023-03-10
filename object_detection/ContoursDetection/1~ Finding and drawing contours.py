"""
:;: What are contours?
::   Contours are the points that is formed by joining all the points on the boundary
        which has same color/intensity on the image
        ** for more accuracy we generally use the gray-scaled images in order to find contours.
     USE: Object-Detection, Shape analysis, Shape detection
"""
import cv2

# Taking images
img = cv2.imread('opencv-logo.png', 1)
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Finding contours of the image  (to find contours we need to have a threshold image of gray-scaled image)
ret, threshImg = cv2.threshold(grayImg, 127, 255, cv2.THRESH_BINARY)  # Here we are applying THRESH_BINARY thresholding

# Finding contours
"""
* findContours() will return 
     1st: A numpy array of contour boundaries
     2nd: A hierarchy (discussed further) """
contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
print("No. of contours = ", len(contours))

# Drawing Contours
rsltImg = cv2.drawContours(img, contours, -1, (209, 0, 255), 4)  # @param: target-image, contour-values, contour-Index(-ve value for all contours, +ve val means particular index of the list of contours), color, thickness
# cv2.drawContours(img, )

# Displaying images
cv2.imshow("Original Image", img)
cv2.imshow("GrayScaledMode", grayImg)
cv2.imshow("Contoured Image", rsltImg)

cv2.waitKey()
cv2.destroyAllWindows()
