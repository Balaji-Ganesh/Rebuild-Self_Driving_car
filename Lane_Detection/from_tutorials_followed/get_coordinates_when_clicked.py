import numpy as np
import cv2

# Some of the mouse click events are..
# events = [i for i in dir(cv2) if 'EVENT' in i]
# print(events)
# print(len(events))  # There are total of 18 mouse click events


def click_events(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:  # If left button is clicked
        print(x, ', ', y)  # Printing co-ordinates on the console
        txtXY = str(x) + ', ' + str(y)  # Getting and Making the x and y coordinates-string so as to print on the image
        font = cv2.FONT_HERSHEY_SIMPLEX  # Taking a font's instance
        cv2.putText(img, txtXY, (x, y), font, .5, (0, 0, 255), 1)
        cv2.imshow('Image', img)
        
    if event == cv2.EVENT_RBUTTONDOWN:  # If Right button is clicked
        #  Getting the channel values at that point where the right button is clicked
        blue = img[x, y, 0]  # Getting the blue channel value at x,y co-ordinate
        green = img[x, y, 1]   # Getting the green channel value at x,y co-ordinate
        red = img[x, y, 2]  # Getting the red channel value at x,y co-ordinate
        print(blue, ', ', green, ',', red)
        txtBGR = str(blue) + ', ' + str(green) + ',' + str(red)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, txtBGR, (x, y), font, .5, (49, 250, 255), 2)
        cv2.imshow('Image', img)


# Creating a black image through numpy.zeros() func
# img = np.zeros([720, 960,  3], np.uint8)
# Creating a instance of color image
img = cv2.imread('road_image.jpg', -1)
print(img.shape)

print(img.shape)
cv2.imshow('Image', img)


# Calling the mouse click event
cv2.setMouseCallback('Image', click_events)  # The paramters passed are : "image" :the name of the window on which the image which we created is showed(means title present on the title bar of the imshow window), and the 2nd parameter is the name of the function for which we made the code to execute that function
cv2.waitKey(0)
cv2.destroyAllWindows()
