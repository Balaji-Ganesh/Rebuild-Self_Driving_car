import cv2
import numpy as np

# Create trackbars..
def dummy(val):
    pass

cv2.namedWindow("Trackbars")
cv2.createTrackbar("lowerHue", "Trackbars", 0, 255, dummy)
cv2.createTrackbar("lowerSat", "Trackbars", 104, 255, dummy)
cv2.createTrackbar("lowerVal", "Trackbars", 88, 255, dummy)
cv2.createTrackbar("higherHue", "Trackbars", 255, 255, dummy)
cv2.createTrackbar("higherSat", "Trackbars", 255, 255, dummy)
cv2.createTrackbar("higherVal", "Trackbars", 255, 255, dummy)

""" Helper functions """
def findObjectByItsColor(capture):
    while True:
        _, frame = capture.read()

        # Change the color space to HSV
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Get the adjusted values from trackbars..
        lwrHue = cv2.getTrackbarPos("lowerHue", "Trackbars")
        lwrSat = cv2.getTrackbarPos("lowerSat", "Trackbars")
        lwrVal = cv2.getTrackbarPos("lowerVal", "Trackbars")
        higherHue = cv2.getTrackbarPos("higherHue", "Trackbars")
        higherSat = cv2.getTrackbarPos("higherSat", "Trackbars")
        higherVal = cv2.getTrackbarPos("higherVal", "Trackbars")

        lowerHSV = np.array([lwrHue, lwrSat, lwrVal])
        higherHSV = np.array([higherHue, higherSat, higherVal])

        # Filter the image..
        maskFrame = cv2.inRange(hsvFrame, lowerHSV, higherHSV)

        # Extract the required color only..
        extractedFrame = cv2.bitwise_and(frame, frame, mask=maskFrame)

        # display
        cv2.imshow("Original feed", frame)
        cv2.imshow("HSV feed", hsvFrame)
        cv2.imshow("Mask feed", maskFrame)
        cv2.putText(extractedFrame, text="Press 'Esc' to capture object's color & exit.", org=(10, 40),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(180, 250, 200), fontScale=0.8, thickness=2)
        cv2.putText(extractedFrame, text=f"lowerHSV: {lowerHSV}, higherHSV: {higherHSV}", org=(10, 70),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    color=(180, 250, 80), fontScale=0.6, thickness=1)
        cv2.imshow("Extracted Feed", extractedFrame)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            return lowerHSV, higherHSV


def detectObjectByItsColor(capture, lowerHSV, higherHSV, minObjectAreaThresh=700):
    while True:
        _, frame = capture.read()

        """ Get the mask of the object ONLY """
        # Change frame's color space.
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Extract the object ONLY by its hsv values
        maskFrame = cv2.inRange(hsvFrame, lowerHSV, higherHSV)
        # Apply threshold, to get a good mask -- by ignoring object's noises..
        _, maskFrame = cv2.threshold(maskFrame, 254, 255, cv2.THRESH_BINARY)
        # Find contours in the mask.
        contours, _ = cv2.findContours(maskFrame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Draw the contours around the object.
        for contour in contours:
            # Apply filter to neglect noises having same color space as target object.
            if cv2.contourArea(contour) > minObjectAreaThresh:
                # Draw the contours..
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, text=f"width: {w}px, height:{h}px, area:{w * h}px", org=(10, 40),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            color=(180, 250, 200), fontScale=0.8, thickness=2)

        # Display results
        cv2.imshow("Object detection", frame)
        cv2.imshow("Mask frame", maskFrame)

        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
    return

""" Driver """
if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)

    # Find the target object to find
    lowerHSV, higherHSV = findObjectByItsColor(capture)

    # Detect object based on its color values (HSV)
    detectObjectByItsColor(capture, lowerHSV, higherHSV, 1000)

    print(f"HSV values - lower: {lowerHSV}, higher: {higherHSV}")

    # resource release
    capture.release()
