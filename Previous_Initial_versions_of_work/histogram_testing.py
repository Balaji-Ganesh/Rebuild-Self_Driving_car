import cv2

img = cv2.imread("lena.jpg")
blue, green, red = cv2.split(img)
cv2.imshow("Image", img)

blue, green, red = cv2.split(img)
print(blue.dtype)
print(list(blue))
print(list(blue.ravel()))
print("Hello world")
cv2.waitKey(0)
