import cv2
import numpy as np

folder_path='captured_checkerboards'

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)

    img_count=0

    while True:
        _, frame = capture.read()
        frameCopy = frame.copy()
        cv2.putText(frameCopy, text="Press 'c' to capture, 'Esc' to exit.", org=(10, 30),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(180, 250, 200), fontScale=0.7, thickness=2)
        cv2.putText(frameCopy, text=f"Images captured: {img_count}", org=(10, 60),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(50, 180, 200), fontScale=1, thickness=2)

        cv2.imshow("Image Capturer", frameCopy)

        if cv2.waitKey(1) & 0xFF == ord('c'):    # when pressed key 'c'..
            # Save the image..
            cv2.imwrite(f"{folder_path}/capture_{img_count}.png", frame)
            img_count+=1

        if cv2.waitKey(1) & 0xFF == 27:  # when pressed key 'Esc'..
            print("Total images captured: ", img_count)
            break

    # resource release.
    capture.release()
    cv2.destroyAllWindows()