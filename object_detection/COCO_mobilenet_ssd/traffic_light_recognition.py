"""
Goal:
    ✓ Detect the traffic light within the image
        - via HAAR cascade classifier.
    ✓ Recognize the traffic light's status of the detected traffic light.
        1. extract the traffic light -- as a RoI from goal_1
        2. Cvt to Grayscale
        3. Apply Gaussian Blur
        4. Find the brightest spot
        5. Approximate spot location

    Idea adopted from: [source](https://zhengludwig.wordpress.com/projects/self-driving-rc-car/)
"""
import cv2
import detector

## Initial setup..
# classifier = cv2.CascadeClassifier('traffic_light.xml')
min_threshold = 125     # min value to proceed for traffic light recognition

def recognizeTrafficLight(img):
    """
    Recognizes the traffic light from the given image.
    @:param img: source image
    :returns the image with annotated traffic light recognition

    Notes
    -----
    Steps:
    """
    # Step-0: Detect traffic light
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # traffic_lights = classifier.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    img, detection_info = detector.detectObjects(img, draw_info=False, detectable_classes=['traffic light'])
    # print(detection_info[0])
    # traffic_light_coords= detection_info[1]

    for _, detection in detection_info:
        # print (detection)
        x, y, w, h = detection
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)
        # Extract the RoI
        roi = gray_img[y + 10:y + h - 10, x + 10:x + w - 10]
        # Apply gaussian blur on the roi
        mask = cv2.GaussianBlur(roi, (25, 25), 0)
        # find the brightest spots
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(mask)

        if maxVal - minVal >= min_threshold:
            print("in process of recognition", end="  :::  ")
            test_img = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
            cv2.circle(test_img, maxLoc, 5, (0, 0, 255), 5)
            cv2.imshow("taffic light", test_img)
            # recognize which light is currently turned on..
            ### !! Proceeding with assumption that, traffic light will be vertical.
            one_light_height = h/3 - 20 # `h` is the height of all 3 lights combined. `-30` -> calibration
            if maxLoc[1] < one_light_height:
                print("red")
            # Neglecting yellow for now.. any way no purpose assigned and also getting flickering issues in detecting.
            elif one_light_height < maxLoc[1] < one_light_height*2:
                print("yellow")
            elif maxLoc[1] > one_light_height*2:
                print("green")
    return img

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)

    while True:
        _, frame = capture.read()

        annotated_img = recognizeTrafficLight(frame)


        # Show the results..
        cv2.imshow("Traffic light recognition", annotated_img)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()
    capture.release()
