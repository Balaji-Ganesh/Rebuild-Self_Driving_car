# From the Lesson_XX~by PySource-Road_lane_detection we've got the values for the Lower and higher HSV values, before heading into this concept, we request you to refer and run it.........
# Importing the required libraries
import cv2
import numpy as np

def dummy(value):  # just a dummy function to give as a parameter for the creation of trackbars
	pass

capture =  cv2.VideoCapture("road_car_view.mp4")
# Creating a Window with a name
cv2.namedWindow("HueSaturationValue Adjuster")
# Creating trackbars for the HueSaturationValue
# For Lower HSV..
cv2.createTrackbar("Lower_Hue", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Hue
cv2.createTrackbar("Lower_Sat", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Saturation
cv2.createTrackbar("Lower_Val", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Value
# For Higher HSV
cv2.createTrackbar("Higher_Hue", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Hue
cv2.createTrackbar("Higher_Sat", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Saturation
cv2.createTrackbar("Higher_Val", "HueSaturationValue Adjuster", 0, 255, dummy)  # for the Value

while True: # Running the infinite loop
	ret, frame = capture.read()
	if ret is False:  # if the video is about to end, then repeat it till the user quits by the appropriate key
		capture = cv2.VideoCapture("road_car_view.mp4")
		continue
	
	colorFrame = frame.copy()   # Taking copy as we need to preserve the original one to draw the lines, and we need the copy as we need to perform operations on it
	# Blurring the image to avoid noises
	blurredFrame = cv2.GaussianBlur(colorFrame, (5, 5), 100)
	# Taking the HSV image as to track the yellow road corners
	hsvImage = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
	
	# Getting the H, S, V values from the trackbars
	# Getting the lower Hue, Saturation, Value values
	lowerHue = cv2.getTrackbarPos("Lower_Hue", "HueSaturationValue Adjuster")
	lowerSat = cv2.getTrackbarPos("Lower_Sat", "HueSaturationValue Adjuster")
	lowerVal = cv2.getTrackbarPos("Lower_Val", "HueSaturationValue Adjuster")
	# Getting the higher Hue, Saturation, Value values
	higherHue = cv2.getTrackbarPos("Higher_Hue", "HueSaturationValue Adjuster")
	higherSat = cv2.getTrackbarPos("Higher_Sat", "HueSaturationValue Adjuster")
	higherVal = cv2.getTrackbarPos("Higher_Val", "HueSaturationValue Adjuster")
	
	lowerHSV_vals = np.array([lowerHue, lowerSat, lowerVal])   # these values are obtained from the previous program and observation
	higherHSV_vals = np.array([higherHue, higherSat, higherVal])
	# find the masked frame
	maskedFrame = cv2.inRange(hsvImage, lowerHSV_vals, higherHSV_vals)
	# Get only the required one
	filteredFrame = cv2.bitwise_and(colorFrame, colorFrame, mask=maskedFrame)
	
	# finding the edges
	cannyEdgedFrame = cv2.Canny(filteredFrame, 100, 100) # Vary the values later based on the output
	# Now detect the lines
	
	lines = cv2.HoughLinesP(cannyEdgedFrame, 1, np.pi/180, 10, minLineLength=50, maxLineGap=200) # Vary the threshold value later based on the output
	# Draw the lines from the data of "lines"
	if lines is not None:
		for line in lines:
			X1, Y1, X2, Y2 = line[0]
			cv2.line(frame, (X1, Y1), (X2, Y2), (0, 255, 0), 3)
	
	cv2.imshow("Final Road Lane", frame)
	cv2.imshow("Filtered Frame", filteredFrame)
	cv2.imshow("Canny Edges frame", cannyEdgedFrame)
	cv2.imshow("Masked Frame", maskedFrame)
	cv2.imshow("Blurred Frame", blurredFrame)
	
	if cv2.waitKey(40) == 27:
		lwr_vals = [lowerHue, lowerSat, lowerVal]
		higher_vals = [higherHue, higherSat, higherVal]
		print("The final values are: ")
		print("Lower HSV values : ", lwr_vals)
		print("Higher HSV values : ", higher_vals)
		break

capture.release()
cv2.destroyAllWindows()