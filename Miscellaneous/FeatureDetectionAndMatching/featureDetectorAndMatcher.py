import cv2
import numpy as np
import imutils

img1 = cv2.imread("query/GATE_2023_CSE_GKP.jpg", 0)
img2 = cv2.imread("test/GATE_2023_CSE_GKP.jpg", 0)
# print(img1.shape)
img2 = imutils.resize(img2, width=395)

orb = cv2.ORB_create(nfeatures=1000)
# Get the key points and descriptors..
key_points1, desc1 = orb.detectAndCompute(img1, None)
key_points2, desc2 = orb.detectAndCompute(img2, None)

# # Draw the key points..
key_pts_img1 = cv2.drawKeypoints(img1, key_points1, None)
key_pts_img2 = cv2.drawKeypoints(img2, key_points1, None)

# Make matches between keypoints..
bruteforce_matcher = cv2.BFMatcher()
knn_matches = bruteforce_matcher.knnMatch(desc1, desc2, k=2)

# Determining the good match.
good_matches = []
for m, n in knn_matches:
    print(m.distance,  0.75 * n.distance)
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

print(len(good_matches))
## draw the matches..,
matched_img = cv2.drawMatchesKnn(img1, key_points1, img2, key_points2, good_matches, None, flags=2)

cv2.imshow("Frame1", img1)
cv2.imshow("Frame_2", img2)
cv2.imshow("Frame1.1", key_pts_img1)
cv2.imshow("Frame_2.1", key_pts_img2)
cv2.imshow("matched_img", matched_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
