# Importing the Required libraries
import cv2
import numpy as np


# Private Required functions
def defineTheCoOrdinates(image, lane_parameters):
	
	if lane_parameters is not None:
		print(lane_parameters)
		slope, intercept = lane_parameters
		# Now Defining the co-ordinates
		Y1 = image.shape[0]  # We'll get image height, it means our line should start from the bottom line of the image
		Y2 = int(Y1 * (3 / 5))  # We'll get the 402, it means our line will extend till the 3/5th of the total image height
		# From the basic concept(y=m*x+c), we can get the X's value as x=(y-c)/m which means x=(Y-Y_intercept)/slope
		X1 = int((Y1 - intercept) / slope)
		X2 = int((Y2 - intercept) / slope)
		return np.array([X1, Y1, X2, Y2])


def drawOptimizedLanes(passedImage, lanes):
	finalImage = passedImage.copy()
	if lanes is not None:  # Check to avoid errors in future for safety
		for lane in lanes:
			X1, Y1, X2, Y2 = lane
			cv2.line(finalImage, (X1, Y1), (X2, Y2), (255, 255, 255), 5)  # Drawing the final lane with white color so as to appear unique in blue-lanes
		return finalImage
	else:
		return passedImage


def findCannyEdges(image):
	# Applying the gaussian blur
	blur_image = cv2.GaussianBlur(image, (5, 5), 10000)  # Vary the sigmaX parameter based on the output
	# Find the edges through a edged detector
	cannyImage = cv2.Canny(blur_image, 50, 150)
	return cannyImage


def findRegionOfInterest(cannyImage):
	height = cannyImage.shape[0]
	#print("Height ", height)
	roadAsPolygon = np.array([[(200, height), (1100, height), (550, 250)]], np.int32)  # Coded as 2D array as cv2.fillPoly() needed multiple-Dimensional arrays..
	# Creating the mask image
	maskImage = np.zeros_like(cannyImage)
	cv2.fillPoly(maskImage, roadAsPolygon, 255)  # Filling the required area with the white color(??? Why only white color? :: Because the white color can accept any color, later in this function defined more clearly)
	# Merging the image
	maskedImage = cv2.bitwise_and(cannyImage, maskImage)  # The mask contains the required area if road with the white color(means 255, which means all 1's in binary, so when we perform Bitwise& operation or image area will be retained as same and rest other made black....if confusing figure it on a paper)
	return maskedImage


def drawTheLinesOnTheImage(orgImage, lines):
	# Creating a copy for safety
	lane_image = np.zeros_like(orgImage)  # Creating the black_image with the orgImage dimensions
	
	# Checking whether lines is empty or not, sometimes they may be empty for some worst cases, to avoid error at that time..
	if lines is not None:
		for line in lines:
			X1, Y1, X2, Y2 = line.reshape(4)  # The parameter 4 defines that reshape the 2D array as 4 columns and unpack and assign respective values to X1, Y1, X2, Y2
			# Now drawing the line
			cv2.line(lane_image, (X1, Y1), (X2, Y2), (255, 0, 0), 3)
	return lane_image  # Finally returning the work done, NOTE: This is a black colored Image


def blendTheImages(lanedImage, colorImage):
	# blendedImg = cv2.addWeighted(src1=lanedImage, alpha=0.8, src2=colorImage, beta=1, gamma=1)
	# What we doing here is blending the original image and the image in which the lines are detected are merged together
	# Here the "lanedImage" pixels are multiplied by 0.8{alpha} so that to make it darker(How darker..?? If the pixel value is less it seems darker, if higher it seems brighter right..??)
	# and the "colorImage" pixels are multiplied by 1{theta}, so to keep it brighter and to visible it more brighter in the background image
	# finally 1{gamma}, just to have a round off value
	return colorImage


def optimizeTheLanesOnImage(alreadyLanedImage, multipleLines):
	# for this we need to optimize the left and right road lane individually.
	"""
	 but how we achieve this...??
	 There is one thing difference between the left-road_lane and the right-road_lane.
	 i.e., The slope of right-road_lane is +ve and the slope of left-road_lane is -ve.
	 But how the slope of right-road_lane is +ve..??
	 :: NOTE: In our image the Yaxis(i.e., rows) increases while coming down and the Xaxis(i.e., columns) increases while moving from left-to right(to have even more clarity, display the image through matplotlib, so we'll get a crystal-clear clarity).
	    * for the right-road_lane which is slant a bit (towards the left), will have a +ve slope as X-axis increased, while the Y-axis increasing..
	    * for the left-road_lane which slant a bit (towards right), will have a -ve slope as Y-axis decrease, while the Y-axis incresed.
	    For more clarity refer down the example...
	    The (X1, Y1) values denote the co-ordinates at the top portion of the image as we come on parsing the image from top-bottom...?? clear
	    and (X2, Y2) values denote the co-ordinates at the bottom portion of the image..

	          0
	          1              /           \
	          2             /             \
	          3            /               \
	          4           /                 \
	          5          /                   \
	          6         /                     \
	          7        /                       \
	          8       /                         \
	          9      /                           \
	         10     /                             \
	          0  1  2  3  4  5  6  7  8  9  10 11 12 13
	    {Basic concept used here is : a general form of a line is y=m*x+c, where m=slope, c=intercept. m=(Y2-Y1)(x2-X1) and c = -m*x+y}
		So now for instance assume the (X1, Y1) and (X2,Y2) denotes the right-road_lane and their values respectively are (9, 1) and (10, 12)
		and to find the slope we use the formula m=(Y2-Y1)/(X2-X1) (theme is rise_over_run, rise means increasing the angle, and run means increse the distance)
		so finally m = (12-1)/(10-9) == 11/1 == 11 (which means +ve).................is it clear..??

		Now for the instance teh (X1, Y1) and (X2, Y2) denote the left-road_lane as (5, 1) and (2, 10)
		finally m = (10-1)/(2-5) = 9/-3 == -3 (which means -ve).......................is it clear..?
	"""
	left_fit = []  # Defining the empty list to have the final averaged left-lane values
	right_fit = []  # Defining the empty list to have the final averaged right-lane values
	if multipleLines is not None:  # If empty, to avoid error..
		for line in multipleLines:
			X1, Y1, X2, Y2 = line.reshape(4)  # Reshaping 2D array into 4 columns and unpacking into 4 different variables
			parameters = np.polyfit(x=(X1, X2), y=(Y1, Y2), deg=1)
			# What np.polyfit() does is it will fit a first degree polynomial which will be a simple function like a line equation(i.e., basically like y=mx+c), it gonna fit the polynomial for our co-ordinates(i.e.,. X1, Y1, X2, and Y2) and return a vector of co-efficients which describes the slope and the y-intercept
			# The parameters are the tuple of X co-ordinates and the tuple of Y-co-ordinates and atlast defines the degree of the equation framed(Here for our case we are dealing with a linear equation(line equation), we've passed '1')
			# The output of polyfit() contains the slope at index 0 and intercept at 1 at each row
			slope = parameters[0]
			intercept = parameters[1]
			# Now we took the values individually, its time to separate the left-road_lane and the right-road_lane
			if slope < 0:  # Then it will be of left-road_lane which we discussed above
				left_fit.append((slope, intercept))
			else:  # Then it will be of right-road_lane
				right_fit.append((slope, intercept))
		
		# Making average of all the lines to have a single precise line
		left_fit_average = np.average(left_fit, axis=0) # The axis denotes that average from top-bottom two fields(slope and intercept)
		right_fit_average = np.average(right_fit, axis=0)
		# Till now we've separated the left-road_lanes and the right-road_lanes, but we need the 2 co-ordinates to draw a line right...?? which we still didn't have
		# print(left_fit_average, "Left ones")
		# print(right_fit_average, "right ones")
		# So getting the co-ordinates
		left_lane = defineTheCoOrdinates(image, left_fit_average)
		right_lane = defineTheCoOrdinates(image, right_fit_average)
		# print(left_lane, "left lane")
		# print(right_lane, "right lane")
		return drawOptimizedLanes(alreadyLanedImage, np.array([left_lane, right_lane]))
	else:
		return alreadyLanedImage


# Actual Code begins...
# Loading the video
capture = cv2.VideoCapture(
	"model_car/road_video_at_11h52m20s_forward_route.avi")
# capture = cv2.VideoCapture("road_video.mp4")
while True:
	ret, frame = capture.read()
	if ret is False:  # looping the video till the user quits explicitly
		print("Video show ended")
		capture = cv2.VideoCapture(
			"model_car/road_video_at_11h52m20s_forward_route.avi")
		continue
	
	# copying the image for safety
	image = frame.copy()
	# Get the edges on the image
	cannyEdgedImage = findCannyEdges(image)
	
	# Define the Region of Interest
	croppedImage = findRegionOfInterest(cannyEdgedImage)
	
	# Drawing the lines on the image
	lines = cv2.HoughLinesP(croppedImage, rho=1, theta=np.pi / 180, threshold=100, minLineLength=40, maxLineGap=50)  # maxLineGap defines that the lines which differ with 40pixels can be merged together to become a single line, minLineGap defines that the lines shorter than the specified length can be ignored, rho defines the accumulator bin's distance resolution and the theta defines the angular resolution in the hough space...
	lanedImage = drawTheLinesOnTheImage(image, lines)  # NOTE: we get a black-colored image..
	# P2N: Till now we've got the lines on the road_lanes, but there are multiple lines, we can't decide our movement based on multiple lines, so we need to optimize them as a single line to take the decision clearly and precisely+accurately right..??
	optimizedLanesImage = optimizeTheLanesOnImage(lanedImage, lines)
	print(optimizedLanesImage)
	cv2.imshow("Optimized lanes image", optimizedLanesImage)
	# Blending the original image and the lines-detected image
	# blendedImage = blendTheImages(frame, optimizedLanesImage)
	# Displaying the image
	# cv2.imshow("Final Road", blendedImage)
	if cv2.waitKey(40) == 27:
		break
	
capture.release()
cv2.destroyAllWindows()