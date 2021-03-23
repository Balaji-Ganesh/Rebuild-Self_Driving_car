import cv2
import numpy as np

"""Functions defined specifically for this module/project"""


def dummy(dummy):
    pass


def initialize_threshold_trackbars():
    """
    This function just initializes the trackbars and the window in which they are to be displayed.
    :return: Nothing

    NOTEs:
    -----
        This function is designed to get called by the function "threshold_filter" only the debug_mode parameter is set to True.
    """
    cv2.namedWindow("Threshold Adjustment", flags=cv2.WINDOW_NORMAL)  # Create a window to place the trackbars..
    cv2.createTrackbar("HUE min", "Threshold Adjustment", 66, 255, dummy)
    cv2.createTrackbar("HUE max", "Threshold Adjustment", 221, 255, dummy)
    cv2.createTrackbar("SAT min", "Threshold Adjustment", 0, 255, dummy)
    cv2.createTrackbar("SAT max", "Threshold Adjustment", 238, 255, dummy)
    cv2.createTrackbar("VAL min", "Threshold Adjustment", 0, 255, dummy)
    cv2.createTrackbar("VAL max", "Threshold Adjustment", 236, 255, dummy)


def threshold_filter(image, debug_mode=False):
    """
    This function takes the image, filter values (i.e., upper and lower boundaries) based on the values
    from the "Threshold adjustment" window and filters the image.
    :param image: Source image on which the filtering is to be applied.
    :return: The filtered image based on threshold adjustment values.
    """
    imgHSV = cv2.cvtColor(src=image, code=cv2.COLOR_BGR2HSV)  # Convert the RGB image to HSV image

    # Get the threshold values only if required (When changed environment), else use the previously tuned values..
    if debug_mode:
        initialize_threshold_trackbars()
        lowerThreshold, upperThreshold = get_thresh_values(imgHSV)
    else:
        lowerThreshold = np.array([66, 0, 0])  # Lower threshold, ignore the pixels below than these pixel values (i.e., convert to black pixels)
        upperThreshold = np.array([221, 238, 236])  # Upper threshold, ignore the pixels above than these pixel values (i.e., convert to black pixels)

    maskImage = cv2.inRange(src=imgHSV, lowerb=lowerThreshold, upperb=upperThreshold)  # Get only the image with lowerb > pixels < upperb   (i.e., as white pixels and rest will be black)

    return maskImage


def get_thresh_values(img_hsv):
    """
    This function gets the trackbar positions from the "Threshold Adjustment" window, packs into lowerboundary and upperboudary values and returns
    :return: lowerboundary_values, upperboundary_values

    NOTEs:
    -----
    This function designed to get called by the "threshold_filter" function, only if the debug_mode parameter for the function is set to True.
    """
    # Get the track bar positions..
    hue_max = cv2.getTrackbarPos("HUE max", "Threshold Adjustment")
    sat_max = cv2.getTrackbarPos("SAT max", "Threshold Adjustment")
    val_max = cv2.getTrackbarPos("VAL max", "Threshold Adjustment")
    hue_min = cv2.getTrackbarPos("HUE min", "Threshold Adjustment")
    sat_min = cv2.getTrackbarPos("SAT min", "Threshold Adjustment")
    val_min = cv2.getTrackbarPos("VAL min", "Threshold Adjustment")

    # Pack these values into arrays..
    lower_boundary, upper_boundary = np.array([hue_min, sat_min, val_min]), np.array([hue_max, sat_max, val_max])
    # print(lower_boundary, upper_boundary)
    # Filter the image based on these trackbar adjustments..
    maskImg = cv2.inRange(src=img_hsv, lowerb=lower_boundary, upperb=upper_boundary)
    cv2.imshow("Threshold Adjustment", maskImg)

    return lower_boundary, upper_boundary


def warpImage(image, warp_points, width, height):
    """
    This function is designed to keep the code simple in the lane_detection module.

    :param image: The image on which the warping should be done.
    :param warp_points: The RegionOfInterest(RoI) points of the image.
    :param width: The actual width of the image to be warped. (Warped image will have this width)
    :param height: The actual height of the image to be warped. (Warped image will have this height)
    :return: Warped image

    NOTEs:
    ----
        This function is designed to get called from the "get_lane_curve" function of Lane_Detection.py
    """
    pts1 = np.float32(warp_points)  # Points of the image from which the RangeOfInterest(RoI) for warping should be taken.
    pts2 = np.float32([(0, 0), (width, 0), (0, height), (width, height)])  # Points how the RoI should get warped (Bird's Eye View)
    transformation_matrix = cv2.getPerspectiveTransform(src=pts1, dst=pts2)  # Creates a transformation matrix, so that by this we can transform the image.
    warped_img = cv2.warpPerspective(src=image, M=transformation_matrix,
                                     dsize=(width, height))  # Now warp the image to RoI with the help of transformation_marix generated earlier.
    return warped_img  # Finally return the warped image.


def initialize_warping_trackbars(initial_values, width=480, height=240):
    """
    This function will initialize the trackbars for the setting of the warping points of RoI.

    :param initial_values:  Previously adjusted values.
    :param width: Actual width of the window.
    :param height: Actual height of the window.
    :return: nothing

    NOTEs:
    -----
    This function is designed to get called from the function "get_lane_curve" from the file of "LineDetection.py"
    """
    cv2.namedWindow("Warping Adjusters")
    cv2.createTrackbar("Width Top", "Warping Adjusters", initial_values[0], width // 2, dummy)
    cv2.createTrackbar("Height Top", "Warping Adjusters", initial_values[1], height, dummy)
    cv2.createTrackbar("Width Bottom", "Warping Adjusters", initial_values[2], height, dummy)
    cv2.createTrackbar("Height Bottom", "Warping Adjusters", initial_values[3], width // 2, dummy)


def get_warping_trackbars_values(width=480, height=240):
    """
    This function will get the values of warping from the window "Warping Adjusters", packs those into co-ordinate points and returns.

    :param width: Actual width of the window.
    :param height: Actual height of the window.
    :return: RoI co-ordinate points that are obtained from the adjusted trackbars.

    NOTEs:
    -----
    This function is designed to get called from the "get_lane_curve" function in the "LineDetection.py" file.
    """
    # Get the values from the adjusted trackbars...
    width_top = cv2.getTrackbarPos("Width Top", "Warping Adjusters")
    height_top = cv2.getTrackbarPos("Height Top", "Warping Adjusters")
    width_bottom = cv2.getTrackbarPos("Width Bottom", "Warping Adjusters")
    height_bottom = cv2.getTrackbarPos("Height Bottom", "Warping Adjusters")

    # Make co-ordinate points, so that we can use them to plot directly on the image
    warp_points = np.float32([(width_top, height_top), (width - width_top, height_top),
                              (width_bottom, height_bottom), (width - width_bottom, height_bottom)])

    return warp_points


def draw_warp_points(image, warp_points):
    for i in range(len(warp_points)):
        cv2.circle(image, (warp_points[i][0], warp_points[i][1]), radius=10, color=(255, 100, 105), thickness=cv2.FILLED)
    cv2.imshow("Warping Adjusters", image)


def getHistogram(img, min_percent=0.1, show_histogram=False, region=1):
    """
    This function finds and plots the  the histogram of the warped image(B/W) with the OpenCV line function(which seems like bins of histogram on zoomout view).

    :param img: The warped B/W image(i.e. Thresholded image).
    :param min_percent: The minimum value below which the sum of each column's values should be neglected. (1 to select all columns  and <1 to select some columns, default value is set to be "0.1")
    :param show_histogram: To display the final histogram value or not. (True for Yes and False for No, Default value is "False"i.e., Histogram result is not shown)
    :param region: How much region(height) of the image should be considered(i.e., height-from some height to the ground [whole width is considered, not to worry about that]).
                    * 1 to select the entire height (from top to bottom)
                    * >1 to select the ((actual_height//(>1 value)) height to bottom. ----- High or low Computation is dependent on this, as this decides how many values in all columns should be considered.
                    * 0 to consider nothing (If decided this, then please don't call this function as it leads to undefined situation(error: DIVIDE_BY_ZERO),----BEWARE).
                  ~ to consider 1/2nd or 1/3rd or 1/4th ... (if passed 2, 3, 4,.... respectively) of the image.
    :return:


    NOTEs and a doubt:
    -----------------
    To understand a bit more clear about Histograms for images, please refer to this video: [1]https://youtu.be/XtdQz6piFpI
        if would like to step more, please refer this [2]https://youtu.be/F9TZb0XBow0 by ProgrammingKnowledge.
    DOUBT

    But sir, if we compare the concept said by this tutorial by Murtaza's workshop and [1]... both seem different right..??
        in [1], its said that the X axis values of the histogram will be the color value (from 0-255, a total of 256 values) (Y axis same i.e., for intensity (terminology from [2]))
                        and
        acc. to the Murtaza sir's explanation..
            we are taking the X axis values to be the index of columns of the thresholded warp image.
    We agree that the concept said by Murtaza sir, gives the solution for the curve value.....BUT, BUT

    but, we are not fully satisfied with this info, please explore a bit more and please clear this doubt.


    """

    # How much height of the image to be considered (from some height to the bottom, refer docstring of this function)
    if region == 1:                                                 # To select the whole image (from top to bottom)
        histogramValues = np.sum(img, axis=0)                       # Get the total sum of the each column (Here as we have 480 pixels as width, we'll sum all the 480 pixels column-wise. (There 240 vertical pixels in each column right..!! as there are 240 rows, these 240 columns are 480(that's what 240 x 480 mean when we say the image dimensions)) )
    else:                                                           # To select the desired height (to bottom)
        histogramValues = np.sum(img[img.shape[0]//region:], axis=0)

    # print(histogramValues)
    max_histogramVal = np.max(histogramValues)                      # Get the max value, so that we can filter out the noise of 10% (or desired) of MAX value in all the column-wise sums. (i.e., Get the max value out of 480 values, as there are 480 summed values)
    # print(np.max(histogramValues), end="\t")
    # print(len(histogramValues))
    min_histogramVal = min_percent * max_histogramVal               # Get the min_histVal, so that we can ignore those columns whose value won't exceed this value (i.e., Getting the filter_value/threshold_value)
    # print(np.max(min_histogramVals))
    indexArray = np.where(histogramValues >= min_histogramVal)      # Get all those indices, whose columns are greater than min/threshold value.
    # print(indexArray, end="\t")
    basePoint = int(np.average(indexArray))                              # Get the center of the
    # print(basePoint)
    cv2.circle(img, center=(int(basePoint), 220), radius=10, color=(100, 30, 150), thickness=cv2.FILLED)
    # cv2.imshow("Center", img)
    if show_histogram:  # only if needed(as when we run on RPi we would not like to burden it much), display else not
        img_histogram = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
        for index, intensity in enumerate(histogramValues):             # For every frame of the video, plot the histogram
            cv2.line(img=img_histogram, pt1=(index, img.shape[0]),      # Once carefully observe the pt1... we can notice that X axis value(abscissa) moves towards right, and Y-value(ordinate) stays constant.
                     pt2=(index, img.shape[0]-intensity//255//region),  # This one too (i.e., pt2), here too 'x', moves as like above, but here for every 'x' 'y' also varies. This 'y' denotes the intensity (i.e., frequency if we talk in terminplogy of histograms) which """varies for every index(column-wise)""", """-> Most important P2N.
                     color=(255, 0, 255), thickness=1)
            # cv2.circle(img_histogram, (basePoint, img.shape[0]), 20, (0, 255, 255), cv2.FILLED)
            # cv2.imshow("Histogram", img_histogram)
        # return basePoint    #, img_histogram   .. Shown in the tutorial, but we are ignoring this and displaying it here itself, as we get problem with no. of returning values.

    return basePoint, img_histogram


"""Custom functions from another projects.."""


def display_concatenated_results(images, image_names, scaleFactor=1):
    """
    :param images: list of images to be concatenated. (as list of list, for clarity refer "RealTimeShapeDetectionUsingContours.py" of "Real_TimeShape_Detection_using_contours" project
    :return: Concatenated images, in the order how the list is passed as  parameter. (as list of list, for clarity refer "RealTimeShapeDetectionUsingContours.py")

    NOTES:
    ------
        (1)
                >>> lst = [[1, 2, 3],[4, 5, 6]]
                >>> len(lst)   # Returns no. of rows..
                2
                >>> len(lst[0])  # returns no. of cols.. P2N--<lst[any_row_number]
                3
        (2)
                Concatenation can be done as
                    -> First concatenating all column images, via cv2.vconcat()
                    -> Then concatenating all the column-concatenated_images horizontally, via cv2.hconcat()
                   or in reverse manner, first concatenating the individual rows, via cv2.hconcat(), then concatenating
                      vertically via cv2.vconcat()
                Below is the implementation for the first approach described above..
    WARNING:
             Doesn't work, when grey scaled images need to be concatenated..!!! SOLVED..!!!! Added some code, now it can do automatically if present B/W images
             Solution:
                >>> img = cv2.imread("Resources/Doraemon.jpg")  # Reading as a color image, can specify flags to read directly as converted image
                ..
                ..
                ..
                >>> grayImg = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)  # Converting to gray scale. Becomes 3 channeled image to 2 channeled image.
                '''
                     A 2 channeled image cannot be concatenated with 3 channeled image.
                     Solution for this: (With help of https://answers.opencv.org/question/175912/how-to-display-multiple-images-in-one-window/)
                '''
                >>> grayImg_3channeled = cv2.cvtColor(src=grayImg, code=cv2.COLOR_GRAY2BGR)  # This makes 2 channeled image to 3 channeled image
                '''
                    Now by this method we can even concatenate the gray-scaled images with colored images.
                ''
                Shortcut:
                >>> grayImg_3channeled = cv2.cvtColor(src=cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY), code=cv2.COLOR_GRAY2BGR)  # First converted to 2 channeled(grayscale) from 3 chanel(color), then converted to 3 channeled(grayscale)
    Phases of execution:
            Phase-1: Validation, whether all the rows has same no. of columns or not
            Phase-2: If phase-1, succeeds.. proceeds for concatenation..
    Upgrade suggestions:
         Resizing the images with a scale factor like in https://pythonexamples.org/python-opencv-cv2-resize-image/
    """
    """PHASE-1: Validation"""
    cols = len(images[0])  # Get the no. of cols in a row, Say first row..can be any row, as all rows must have same no. of cols
    for row_count in range(len(images)):
        if len(images[row_count]) is not cols:
            # print("Received {0}Rows and {1}columns".format(images.shape[0], images.shape[1]))
            print("ERROR: All the rows must have same no. of columns, Please check the list passed..!!")
            return None
    """PHASE-2: Concatenating Images"""
    temp_vertical_concatenated_images = []
    for col_idx in range(len(images[0])):
        temp_row_images = []  # Empty each time after concatenating..!! MUST, else "cv2.error: OpenCV(4.2.0) C:\projects\opencv-python\opencv\modules\core\src\matrix_operations.cpp:68: error: (-215:Assertion failed) src[i].dims <= 2 && src[i].rows == src[0].rows && src[i].type() == src[0].type() in function 'cv::hconcat'" error
        for row_idx in range(len(images)):
            # If the image is of gray scale(2 channels).. convert to BGR scale (2 channels) as for a gray_image len(shape) gives 2 where as for a color image it results 3
            if len(images[row_idx][col_idx].shape) == 2:
                # print("grayscale image: at", (row_idx, col_idx))
                # print("Converting to BGR format")
                images[row_idx][col_idx] = cv2.cvtColor(src=images[row_idx][col_idx], code=cv2.COLOR_GRAY2BGR)
            # Add the labels..
            cv2.rectangle(img=images[row_idx][col_idx], pt1=(0, 0), pt2=(int(len(image_names[row_idx][col_idx]) * 15), 50), color=(255, 255, 255), thickness=cv2.FILLED)
            cv2.putText(img=images[row_idx][col_idx], text=image_names[row_idx][col_idx], org=(5, 30), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.8, color=(255, 0, 255), thickness=2)
            # Resizing..
            images[row_idx][col_idx] = cv2.resize(src=images[row_idx][col_idx], dsize=(0, 0), fx=scaleFactor, fy=scaleFactor)
            temp_row_images.append(images[row_idx][col_idx])
        temp_vertical_concatenated_images.append(cv2.vconcat(temp_row_images))
    return cv2.hconcat(temp_vertical_concatenated_images)


def draw_rounded_rectangle(img, top_left, bottom_right, radius=10, thickness=cv2.FILLED, fill_color=(100, 0, 200), alpha=1):
    """
    This function draws the rounded rectangle with the given dimensions and radius of the corners.

    :param img:             The image on which the rounded rectangle is to be drawn.
    :param top_left:        Top-left co-ordinate of the [normal]rectangle.
    :param bottom_right:    Bottom-right corner co-ordinate fot the [normal] rectangle.
    :param radius:          Radius of the corners (in terms of pixels).
    :param thickness:       Thickness of the rounded rectangle. (If -1 or cv2.FILLED, rounded rectangle will be filled else it denotes the thickness of the rounded rectangle)
    :param fill_color:      Color of the Rounded rectangle
    :param alpha:           Dimming the image (1 shows no effect, to have effect it must be b/w 0>alpha<1)

    :return:                The image on which the rounded rectangle is drawn.

    NOTEs
    -----
    Conventions used:
        As per the mathematics:
            +theta  : Anti-clockwise direction
            -theta  : Clock-wise direction

        *we referred theta for angle parameter for cv2.ellipse function.

    ---- Coded on 17th September, 2020 for the project of "Autonomous self Driving Car".-------
    """
    # Adjust the co-ordinates as per the radius
    x1, y1 = tuple(np.array(top_left) + radius)  # Top_left is adjusted by ""adding"" with radius..          So, later while using x1 and y1 they are ""subtracted"" with radius wherever necessary..
    x2, y2 = tuple(np.array(bottom_right) - radius)  # Bottom_right is adjusted by ""subtracting"" with radius.. So, later while using x2 and y2 they are ""added"" with radius wherever necessary.

    """First draw the quarter circles at each edge, whole becomes the rounded rectangle along with drawn lines"""
    # Top-left
    cv2.ellipse(img=img, center=(x1, y1), axes=(radius, radius), angle=90, startAngle=90, endAngle=180, color=fill_color, thickness=thickness)
    # Top-right
    cv2.ellipse(img=img, center=(x2, y1), axes=(radius, radius), angle=-90, startAngle=90, endAngle=0, color=fill_color, thickness=thickness)
    # Bottom-left
    cv2.ellipse(img=img, center=(x1, y2), axes=(radius, radius), angle=-90, startAngle=180, endAngle=270, color=fill_color, thickness=thickness)
    # Bottom-right
    cv2.ellipse(img=img, center=(x2, y2), axes=(radius, radius), angle=90, startAngle=0, endAngle=-90, color=fill_color, thickness=thickness)

    """Now draw the sides ..."""
    # This special filling is only required for the lines setup, as for ellipses it can filled by the command(i.e., not required for rhe arc's of the rounded rectangle).
    if thickness == cv2.FILLED or thickness < 0:
        cv2.rectangle(img=img, pt1=(x1 - radius, y1), pt2=(x2 + radius, y2), color=fill_color, thickness=cv2.FILLED)
        cv2.rectangle(img=img, pt1=(x1, y1 - radius), pt2=(x2, y2 + radius), color=fill_color, thickness=cv2.FILLED)
        thickness = -thickness  # This is required(i.e., -ve to +ve).. as we can't pass -ve value for the thickness of the cv2.line as line is not a 2D shape which we can fill a color...
        # print(thickness)
    # Top_side
    cv2.line(img=img, pt1=(x1, y1 - radius), pt2=(x2, y1 - radius), color=fill_color, thickness=thickness)
    # Bottom_side
    cv2.line(img=img, pt1=(x1, y2 + radius), pt2=(x2, y2 + radius), color=fill_color, thickness=thickness)
    # left_side
    cv2.line(img=img, pt1=(x1 - radius, y1), pt2=(x1 - radius, y2), color=fill_color, thickness=thickness)
    # right_side
    cv2.line(img=img, pt1=(x2 + radius, y1), pt2=(x2 + radius, y2), color=fill_color, thickness=thickness)
    # cv2.line(img=img, pt)

    """Linear Blending of the image [src: https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html]"""
    # g(x)=(1−α)f0(x)+αf1(x) ie., dst=α⋅src1+β⋅src2+γ    here "α" for alpha that denotes the brightness, "β" means (1-α) finally "γ" an additional parameter (normally this is set to 0.0)
    image = np.ones_like(img)  # A full white image with same dimensions of the passed image.
    dst = cv2.addWeighted(src1=img, alpha=alpha, src2=image, beta=(1 - alpha), gamma=0.0)
    # cv2.imshow("Design image Weighted", dst)
    # cv2.imshow("Design image", img)
    return dst


def text_in_rounded_rectangle(img):
    pass


def draw_curve_scale(img, curve_measure=0):
    height, width, depth = img.shape
    scale_center = (width // 2, (height // 2) + int(0.3 * height))
    (x0, y0) = scale_center

    # For Linear Blending of some designs (Scale and Zooming lens)
    overlay = img.copy()
    # overlay = draw_rounded_rectangle(img=overlay, top_left=(x0 - width // 2 + 30, y0 - 30), bottom_right=(x0 + width // 2 - 30, y0 + 30), radius=10, thickness=cv2.FILLED, fill_color=(25, 0, 0))
    # Draw all the scales..
    for x in range(x0 - width // 2 + 40, x0 + width // 2 - 40 + 1, 10):
        cv2.line(img=img, pt1=(x, y0 + 15), pt2=(x, y0 - 15), color=(200, 200, 0), thickness=2)
        # print(x)

    # Draw a perpendicular line joining all these above drawn lines..
    cv2.line(img=img, pt1=(x0 - width // 2 + 35, y0), pt2=(x0 + width // 2 - 35, y0), color=(200, 200, 0), thickness=2)

    # Draw the scale_center
    cv2.line(img=img, pt1=(x0, y0 - 20), pt2=(x0, y0 + 20), color=(255, 0, 0), thickness=4)

    # Draw a rounded box and fill the box (on overlay) around the scale....
    overlay = draw_rounded_rectangle(img=overlay, top_left=(x0 - width // 2 + 30, y0 - 30), bottom_right=(x0 + width // 2 - 30, y0 + 30), radius=10, thickness=cv2.FILLED, fill_color=(25, 0, 0))
    img = draw_rounded_rectangle(img=img, top_left=(x0 - width // 2 + 30, y0 - 30), bottom_right=(x0 + width // 2 - 30, y0 + 30), radius=10, thickness=1, fill_color=(0, 0, 255))

    """ Draw the curve measure as a line (!! curve_measure is just a value that tells the value on scale from origin..)"""
    curve_measure = curve_measure * 10
    # Vertical line denoting the curve_measure..
    cv2.line(img=img, pt1=(x0 + curve_measure, y0 - 30), pt2=(x0 + curve_measure, y0 + 25), color=(0, 255, 0), thickness=3)
    # Horizontal line from the origin to the curve_measure...
    cv2.line(img=img, pt1=(x0 + 2, y0), pt2=(x0 + curve_measure, y0), color=(0, 200, 100), thickness=3)

    # Add a zooming lens over the measure_scale-value like we get when we type on a android keyboard..
    cv2.circle(img, center=(x0 + curve_measure, y0 - 60), radius=30, color=(0, 255, 0), thickness=3)
    # Fill the lens with blue color and add it to the original image..
    alpha = 0.1
    cv2.circle(overlay, center=(x0 + curve_measure, y0 - 60), radius=30 - 2, color=(255, 100, 50), thickness=-1)
    img = cv2.addWeighted(src1=overlay, alpha=alpha, src2=img.copy(), beta=(1 - alpha), gamma=0.0)
    # Add the text in the zooming lens drawn over the circle on top of measure_value...
    cv2.putText(img=img, text=str(curve_measure // 10), org=(x0 + curve_measure - len(str(curve_measure)) * 10 + 10, y0 - 60 + 13),
                fontFace=cv2.FONT_HERSHEY_DUPLEX, color=(255, 0, 0), fontScale=1.1, thickness=2)
    # cv2.putText()
    # cv2.imshow("Scale", img)
    return img


def design_template(img, curve_measure):
    # Get the measurements and calculate the center of the image..
    height, width, channels = img.shape
    x0, y0 = width // 2, height // 2
    print(x0, y0)

    # Overlay's .. later to blend them for special design..
    overlay = img.copy()
    alpha = 0.4

    # "Text" that tells the status..
    text = ""
    if curve_measure > 0:
        text = "Moving RIGHT.."
    elif curve_measure < 0:
        text = "Moving LEFT.."
    else:
        text = "Moving STRAIGHT.."

    # Draw a Bounding box to display the above text..
    text_length = len(text)
    # Draw a border_bounded_box (very thin border)
    img = draw_rounded_rectangle(img=img, top_left=(x0-text_length*10-10, y0-int(height/2.3)), bottom_right=(x0+text_length*10+10, y0-int(height/2.3)+50), radius=10, thickness=1, fill_color=(0, 0, 255))
    # Draw the the background rectangle for extra effect (as it will be overlayed)
    overlay = draw_rounded_rectangle(img=overlay, top_left=(x0 - text_length * 10 - 10, y0-int(height/2.3)), bottom_right=(x0 + text_length * 10 + 10,y0-int(height/2.3)+50), radius=10, thickness=cv2.FILLED, fill_color=(0, 100, 200))
    img = cv2.addWeighted(src1=overlay, alpha=alpha, src2=img, beta=1 - alpha, gamma=0.0)
    # turn-2
    overlay = draw_rounded_rectangle(img=img, top_left=(x0-text_length*10, y0-int(height/2.3)), bottom_right=(x0+text_length*10, y0-int(height/2.3)+50), radius=10, thickness=cv2.FILLED, fill_color=(0, 0, 200))
    img = cv2.addWeighted(src1=overlay, alpha=alpha+0.2, src2=img, beta=1-alpha+0.2, gamma=0.0)

    img = draw_curve_scale(img=img, curve_measure=curve_measure)



    # place the text in the bounded box..
    img = cv2.putText(img=img, text=text, org=(x0-text_length*10+35, y0-int(height/2.3)+50-15), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    cv2.imshow("Final result on Design Frame", img)
