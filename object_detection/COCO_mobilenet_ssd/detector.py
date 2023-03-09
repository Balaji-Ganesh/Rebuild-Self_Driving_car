import cv2

# img = cv2.imread('lena.jpg')    # to work on saved image
capture = cv2.VideoCapture(0)   # to work on real-time feed.

# Load all the classes
classes = []    # to store all the classes
with open('labels.txt', 'rt') as file:
    classes = file.read().rstrip('\n').split('\n')
# print(classes)

# Configuration of the mobilenet model
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath, configPath)
# configurations as per docmentation --- find link
net.setInputSize(320, 320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

while True:
    _, img = capture.read()     # read the frames from camera
    # Send image to model to get prediction.
    classIds, confidences, bounding_boxes = net.detect(
        img, confThreshold=0.6)  # consider if confidence is >50%

    if len(bounding_boxes) > 0:   # When detected any object..
        print("#objects detected: ", len(bounding_boxes))
        # Overlay the prediction on the image..
        for classId, confidence, bbox in zip(classIds, confidences, bounding_boxes):
            cv2.rectangle(img, bbox, color=(0, 255, 0), thickness=2)
            print("id: ", classId)
            cv2.putText(img, classes[classId-1].upper(),
                        (bbox[0]+10, bbox[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    # Display output
    cv2.imshow('Output', img)
    cv2.waitKey(1)
