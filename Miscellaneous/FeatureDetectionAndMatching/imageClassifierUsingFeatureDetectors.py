import cv2
import numpy as np
import imutils
import os

def loadQueryImages(path):
    """
    Loads the images in the given path.
    :param path: the path, where the query images are located.
    :return: 2 lists - list of images in passed query path, another of their respective classes (their image names)
    """
    # Detect the images in the given path
    path_contents = os.listdir(path=path)
    print("Total images detected: ", len(path_contents))

    # Load all the images in query_path
    query_images = []  # a list to store all the query images
    query_classes = []  # a list to store corresponding  query image name (as a class)
    for img_name in path_contents:
        query_images.append(cv2.imread(f'{query_path}/{img_name}', 0)) # Load as gray scale
        query_classes.append(img_name[:-4])  # or `os.path.splitext(img_name)[0]`
    print("Classes loaded: ", query_classes)
    return query_images, query_classes

def findDescriptorsOfQueryImages(query_images):
    """
    Computes the descriptors of the query images
    :param query_images: list of query images (in grayscale)
    :return: list of (corresponding) descriptors of each query image.
    """
    # Get the key points and descriptors..
    query_descriptors = []
    for image in query_images:
        key_points, descriptor = orb.detectAndCompute(image, None)
        query_descriptors.append(descriptor)
    return query_descriptors

def findClass(image, descriptors, min_match_threshold=15):
    """
    finds the matching class_id based on given image and descriptors
    :param image: gray scaled image
    :param descriptors: list of descriptors of query images, computed earlier.
    :param min_match_threshold: threshold, that good_match's length has to meet
    :return: index of class matched. If not found, -1.
    """
    # Find descriptor of passed image..
    key_points, image_descriptor = orb.detectAndCompute(image, None)

    # Make matches between descriptors to find the class..
    bruteforce_matcher = cv2.BFMatcher()
    matches_lengths = []
    try:
        for descriptor in descriptors:
            knn_matches = bruteforce_matcher.knnMatch(image_descriptor, descriptor, k=2)

            """ Determining the good match. """
            good_matches = []
            for m, n in knn_matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
            # Take the length of good match
            matches_lengths.append(len(good_matches))
    except Exception as e:
        print("[ERROR] An exception caused.\nDetailed report:\n", e)

    # Figure out the max good match's length
    if len(matches_lengths) != 0 and max(matches_lengths) > min_match_threshold:
        return matches_lengths.index(max(matches_lengths))
    else:
        return -1   # to indicate - nothing matched

if __name__ == '__main__':
    """ Initial setup """
    # Camera setup
    capture = cv2.VideoCapture(0)
    capture.set(3, 640)
    capture.set(4, 480)

    # Environment setup
    query_path = 'query'  # Path where the base images are to be looked up
    orb = cv2.ORB_create(nfeatures=1000)

    """ Get into work """
    # Preprocessing..
    query_images, query_classes = loadQueryImages(query_path)
    query_descriptors = findDescriptorsOfQueryImages(query_images)

    # Get! Set! Go..!!!
    while True:
        _, frame = capture.read()   # Get the feed from camera.
        # Determine the class
        idx = findClass(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), query_descriptors)
        """ Show the results.. """
        # Make annotations
        cv2.putText(frame, text='Class: ', org=(25, 25), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(255, 255, 255), fontScale=0.8, thickness=1)
        if idx != -1:
            cv2.putText(frame, text=query_classes[idx], org=(105, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 255, 0), fontScale=0.9, thickness=2)
        else:
            cv2.putText(frame, text="Not Found :(", org=(105, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(0, 0, 255), fontScale=0.9, thickness=2)

        cv2.imshow("~~ Classifier ~~", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Resources release
    capture.release()
    cv2.destroyAllWindows()

"""
Notes
-----
    - image need to be 2D
    - should have good texture.

References
----------
[1] https://youtu.be/nnH55-zD38I

Report
------
Final report on 11th March, 2023 - Sat
- Working fine..!!
"""
