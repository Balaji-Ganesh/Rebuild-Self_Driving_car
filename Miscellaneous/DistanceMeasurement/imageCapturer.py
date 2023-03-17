import cv2
import numpy as np

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(3, 480)

    while True:
        _, frame = capture.read()
        frameCopy = frame.copy()
        cv2.putText(frameCopy, text="Press 'c' to capture and exit.", org=(10, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(180, 250, 200), fontScale=1, thickness=2)
        cv2.imshow("Image Capturer", frameCopy)

        if cv2.waitKey(1) & 0xFF == ord('c'):    # when pressed key 'c'..
            # Save the image..
            cv2.imwrite("capture.png", frame)
            # exit..
            break

    # resource release.
    capture.release()
    cv2.destroyAllWindows()