"""
An experiment.. in knowing how calibration works.. and where need to apply

guide: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html, jump to section of "Undistortion"
How it is working.. ( do after reading the above article)
    - run the script..
    - 3 windows will popup -- analyze those by knowledge of above article. 


"""

import numpy as np
import cv2

# Load the stored calibration matrices..
mtx = np.loadtxt('captured_checkerboards/cameraMatrix.txt', dtype='float', delimiter=',')
dist = np.loadtxt('captured_checkerboards/cameraDistortion.txt', dtype='float', delimiter=',')

img = cv2.imread('captured_checkerboards/capture_0.png')
print(img.shape)
h, w, ch = img.shape
new_cam_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

undistorted = cv2.undistort(img, mtx, dist, None, new_cam_mtx)

# crop.
x, y, w, h, = roi
cropped = undistorted[y:y+h, x:x+w]

cv2.imshow("original image", img)
cv2.imshow("undistorted image", undistorted)
cv2.imshow("Cropped image", cropped)

cv2.waitKey(0)
cv2.destroyAllWindows()
