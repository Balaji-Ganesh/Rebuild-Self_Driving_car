import cv2
import numpy as np

def empty(value):
    pass

cv2.namedWindow("Parameters")
cv2.resizeWindow("Parameters", 640, 120)
cv2.createTrackbar("thresh1", "Parameters", 23, 255, empty)
cv2.createTrackbar("thresh2", "Parameters", 22, 255, empty)
cv2.createTrackbar("minAreaToConsider", "Parameters", 1000, 30000, empty)

def preprocess(img):
    blurImg = cv2.GaussianBlur(img, (7, 7), 1)          # Apply blur
    grayImg = cv2.cvtColor(blurImg, cv2.COLOR_BGR2GRAY)  # Cvt to gray scale

    # Detect edges..
    thresh1 = cv2.getTrackbarPos("thresh1", "Parameters")
    thresh2 = cv2.getTrackbarPos("thresh2", "Parameters")
    cannyImg = cv2.Canny(grayImg, thresh1, thresh2)

    # Spread out the edges detected -- multiple lines become 1.
    kernel = np.ones((5, 5))
    dilatedImg = cv2.dilate(cannyImg, kernel, iterations=1)
  
    return dilatedImg


def getContours(preprocessedImg, imageToDraw):
    # find the contours..
    contours, hierarchy = cv2.findContours(preprocessedImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      
    # Filter the unwanted noise
    for contour in contours:
        # Get the area of contour -- to filter noise.
        contourArea = cv2.contourArea(contour)
        minArea = cv2.getTrackbarPos("minAreaToConsider", "Parameters")
        # Apply filter
        if contourArea >= minArea:
            # print((contourArea, minArea))
            cv2.drawContours(imageToDraw, contours, -1, (255, 0, 255), 7)
            # Detecting shapes based on corners..
            perimeter = cv2.arcLength(contour, True)    # is_contour_closed: True
            # Approximate shape..
            approx =cv2.approxPolyDP(contour, 0.02*perimeter, True)
            # print(len(approx))
            # Draw bounding boxes -- to highlight
            x,y, w, h = cv2.boundingRect(approx)
            cv2.rectangle(imageToDraw, (x, y), (x+w, y+h), (0, 255, 0), 5)
            cv2.putText(imageToDraw, "#Points: "+str(len(approx)) ,(x+w+20, y+20), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)
            cv2.putText(imageToDraw, "Area: "+str(int(contourArea)) ,(x+w+20, y+50), cv2.FONT_HERSHEY_COMPLEX, .7, (0, 255, 0), 2)

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)  # Camera frame width
    capture.set(4, 480)  # Camera frame height

    while True:
        _, frame = capture.read()
        imageToDrawContours = frame.copy()
        # Perform preprocessing
        img = preprocess(frame)
        # Get and draw contours..
        getContours(img, imageToDrawContours)

        cv2.imshow("Preprocessed", img)
        cv2.imshow("Drawn Contours", imageToDrawContours)
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
