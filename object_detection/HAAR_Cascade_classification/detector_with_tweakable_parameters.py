import cv2

# Setup..
cascade_path = 'haarcascade_frontalface_default.xml'   # pass the path of cascade file.
frameWidth = 640
frameHeight = 480
drawingBoxColor = (180, 0, 255)  # Pink
cascadeObjectName = 'Person face'  # name of object, on which the cascade is trained on 

def dummy(val):
    pass

# window for tweaking parameters..
windowName = "Output with trackbars"
cv2.namedWindow(windowName)
cv2.resizeWindow(windowName, frameWidth, frameHeight+100)
cv2.createTrackbar("scale", windowName, 400, 1000, dummy)
cv2.createTrackbar("neighbors", windowName, 4, 200, dummy)
cv2.createTrackbar("min area", windowName, 43400, 100000, dummy)
cv2.createTrackbar("brightness", windowName, 180, 255, dummy)

# Load the cascade..
cascade = cv2.CascadeClassifier(cascade_path)

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, frameWidth)
    capture.set(4, frameHeight)

    while True:
        cameraBrightness = cv2.getTrackbarPos("Brightness", windowName) # Get the tweaked brightness value
        # capture.set(10, cameraBrightness)   # set the brightness adjusted
        # Get the frame from camera capture
        _, frame = capture.read()
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # cvt to gray scale -- for better computation

        # Take and set the detection parameters..
        scale = 1 + (cv2.getTrackbarPos("scale", windowName)/1000)# Lower the scale value - higher the computation power (and better results)
        neighbors = cv2.getTrackbarPos("neighbors", windowName)
        # Perform detection of object..
        detectedObjects = cascade.detectMultiScale(grayFrame, scaleFactor=1.1, minNeighbors=4)

        # Draw the detection on the image..
        for (x, y, w, h) in detectedObjects:    
            object_area = w * h
            minArea = cv2.getTrackbarPos("min area", windowName)
            # Filtration..
            if object_area > minArea:
                roi = frame[y:y+h, x:x+w]
                cv2.rectangle(frame, (x, y), (x+w, y+h), drawingBoxColor, 3)
                cv2.putText(img=frame, text=cascadeObjectName.upper(), org=(x, y-10), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=drawingBoxColor, fontScale=0.5, thickness=1)
                cv2.imshow("ROI", roi)
        cv2.imshow(windowName, frame)

        if cv2.waitKey(1) == 27 or 0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()