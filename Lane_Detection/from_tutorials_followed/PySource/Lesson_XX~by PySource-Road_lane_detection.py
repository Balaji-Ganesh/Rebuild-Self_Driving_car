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
while True:  # Running the infinite loop
	ret, colorFrame = capture.read()
	if ret is False:  # What this does is... When the video is finished, it loads it again, so by this we can keep on running(it is as like as being in a infinite loop)
		capture = cv2.VideoCapture("road_car_view.mp4")
		continue
	
	grayFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)  # converting into gray-scale as processing can be done easily {just for the purpose of detecting the white middle broken lines}
	hsvFrame = cv2.cvtColor(colorFrame, cv2.COLOR_BGR2HSV)  # Converting into HSV as to detect the side's yellow color lanes
	
	# Getting the H, S, V values from the trackbars
	# Getting the lower Hue, Saturation, Value values
	lowerHue = cv2.getTrackbarPos("Lower_Hue", "HueSaturationValue Adjuster")
	lowerSat = cv2.getTrackbarPos("Lower_Sat", "HueSaturationValue Adjuster")
	lowerVal = cv2.getTrackbarPos("Lower_Val", "HueSaturationValue Adjuster")
	# Getting the higher Hue, Saturation, Value values
	higherHue = cv2.getTrackbarPos("Higher_Hue", "HueSaturationValue Adjuster")
	higherSat = cv2.getTrackbarPos("Higher_Sat", "HueSaturationValue Adjuster")
	higherVal = cv2.getTrackbarPos("Higher_Val", "HueSaturationValue Adjuster")
	
	lower_hsv = np.array([lowerHue, lowerSat, lowerVal])
	higher_hsv = np.array([higherHue, higherSat, higherVal])
	
	# Getting the Road borders
	maskedImage = cv2.inRange(hsvFrame, lower_hsv, higher_hsv)
	# Doing the bitwise_and operation, so as to keep only which is required and black the one which is not required
	resultantImage = cv2.bitwise_and(colorFrame, colorFrame, mask=maskedImage)
	
	# Displaying the Results
	cv2.imshow("Original Frame", colorFrame)
	cv2.imshow("Final Detected Image", resultantImage)
	cv2.imshow("Masked image", maskedImage)
	cv2.imshow("HueSaturationValued image", hsvFrame)
	
	if cv2.waitKey(40) == 27:
		lwr_vals = [lowerHue, lowerSat, lowerVal]
		higher_vals = [higherHue, higherSat, higherVal]
		print("The final values are: ")
		print("Lower HSV values : ",lwr_vals)
		print("Higher HSV values : ", higher_vals)
		break

capture.release()
cv2.destroyAllWindows()