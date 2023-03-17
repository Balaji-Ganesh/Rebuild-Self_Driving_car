# Importing the Required libraries
import cv2
import numpy as np

# Loading the image
image = cv2.imread("Rectangle_block.png")

# Applying EdgeDetection
cannyEdges = cv2.Canny(image, 50, 10)

# Drawing the lines on the image
lines = cv2.HoughLinesP(cannyEdges, 1, np.pi/180, 10, maxLineGap=150)
if lines is not None:  # Some times we may not get lines, so to avoid error at that time...
	for line in lines:
		X1, Y1, X2, Y2 = line[0]
		cv2.line(image, (X1, Y1), (X2, Y2), (0, 255, 0), 3)

# Displaying the image
cv2.imshow("Image", image)
cv2.imshow("Canne Edges", cannyEdges)
cv2.waitKey()
cv2.destroyAllWindows()
