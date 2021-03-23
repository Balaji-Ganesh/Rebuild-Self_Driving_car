import cv2
import numpy as np


def draw_rounded_rectangle(img, top_left, bottom_right, radius=10, thickness=cv2.FILLED, fill_color=(100, 0, 200), alpha=1.0):
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
    * A small note for the cv2.ellipse(), axes means (major_axis_length, minor_axis_length)
        ref: https://docs.opencv.org/master/dc/da5/tutorial_py_drawing_functions.html
    * Reference: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
        *we referred theta for angle parameter for cv2.ellipse function.

    ---- Coded on 17th September, 2020 for the project of "Autonomous self Driving Car".-------

    Upgradation Sugggestions:
        making this function alone do the major work of rounded_rectangles used in the "design_template" function.
            i.e., drawing a light colored rectangle around the main rounded rectangle..
                        (And also giving the which sides it should get fill[in the form of a list like : [1, 1, 1, 1] represents for ht e[top, right, bottom, left],
                         default will be all sides(ie., [1, 1, 1, 1], and custom like [1, 0, 1, 1]))
            if would like to have an example (we hard-coded it but make it dynamic when going to implement in this funtion)
                Refer the display of "Final result on Design Frame" window of *this project

        ---------------------------------------Suggestion written on 22nd September, 2020 ~ Tuesday while working for the project of *"Autonomous Self driving car"
    """
    # Adjust the co-ordinates as per the radius
    x1, y1 = tuple(np.array(top_left) + radius)  # Top_left is adjusted by ""adding"" with radius..          So, later while using x1 and y1 they are ""subtracted"" with radius wherever necessary..
    x2, y2 = tuple(np.array(bottom_right) - radius)  # Bottom_right is adjusted by ""subtracting"" with radius.. So, later while using x2 and y2 they are ""added"" with radius wherever necessary.

    overlay = img.copy()  # A full white image with same dimensions of the passed image.
    output = img.copy()

    """First draw the quarter circles at each edge, whole becomes the rounded rectangle along with drawn lines"""
    #
    # Top-left
    cv2.ellipse(img=overlay, center=(x1, y1), axes=(radius, radius), angle=90, startAngle=90, endAngle=180, color=fill_color, thickness=thickness)
    # Top-right
    cv2.ellipse(img=overlay, center=(x2, y1), axes=(radius, radius), angle=-90, startAngle=90, endAngle=0, color=fill_color, thickness=thickness)
    # Bottom-left
    cv2.ellipse(img=overlay, center=(x1, y2), axes=(radius, radius), angle=-90, startAngle=180, endAngle=270, color=fill_color, thickness=thickness)
    # Bottom-right
    cv2.ellipse(img=overlay, center=(x2, y2), axes=(radius, radius), angle=90, startAngle=0, endAngle=-90, color=fill_color, thickness=thickness)

    """Now draw the sides ..."""
    # This special filling is only required for the lines setup, as for ellipses it can filled by the command(i.e., not required for rhe arc's of the rounded rectangle).
    if thickness == cv2.FILLED or thickness < 0:
        cv2.rectangle(img=overlay, pt1=(x1 - radius, y1), pt2=(x2 + radius, y2), color=fill_color, thickness=cv2.FILLED)
        cv2.rectangle(img=overlay, pt1=(x1, y1 - radius), pt2=(x2, y2 + radius), color=fill_color, thickness=cv2.FILLED)
        thickness = -thickness  # This is required(i.e., -ve to +ve).. as we can't pass -ve value for the thickness of the cv2.line as line is not a 2D shape which we can fill a color...
        print(thickness)
    # Top_side
    cv2.line(img=overlay, pt1=(x1, y1 - radius), pt2=(x2, y1 - radius), color=fill_color, thickness=thickness)
    # Bottom_side
    cv2.line(img=overlay, pt1=(x1, y2 + radius), pt2=(x2, y2 + radius), color=fill_color, thickness=thickness)
    # left_side
    cv2.line(img=overlay, pt1=(x1 - radius, y1), pt2=(x1 - radius, y2), color=fill_color, thickness=thickness)
    # right_side
    cv2.line(img=overlay, pt1=(x2 + radius, y1), pt2=(x2 + radius, y2), color=fill_color, thickness=thickness)
    # cv2.line(img=img, pt)

    """Linear Blending of the image [src: https://docs.opencv.org/3.4/d5/dc4/tutorial_adding_images.html]"""
    # g(x)=(1−α)f0(x)+αf1(x) ie., dst=α⋅src1+β⋅src2+γ    here "α" for alpha that denotes the brightness, "β" means (1-α) finally "γ" an additional parameter (normally this is set to 0.0)

    # print(alpha)
    # cv2.imshow("overlay", overlay)
    dst = cv2.addWeighted(src1=overlay, alpha=alpha, src2=output, beta=(1 - alpha), gamma=0.0)
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
        print(x)

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
    cv2.imshow("Scale", img)
    return img


def design_template(img, curve_measure):
    # Get the measurements and calculate the center of the image..
    height, width, channels = img.shape
    x0, y0 = width // 2, height // 2

    # Overlay's .. later to blend them for special design..
    overlay = img.copy()
    alpha=0.5

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
    img = draw_rounded_rectangle(img=img, top_left=(x0-text_length*10-10, y0-180), bottom_right=(x0+text_length*10+10, y0-110+10), radius=10, thickness=1, fill_color=(0, 0, 255))
    overlay = draw_rounded_rectangle(img=img, top_left=(x0 - text_length * 10 - 10, y0 - 180), bottom_right=(x0 + text_length * 10 + 10, y0 - 110 + 10), radius=10, thickness=cv2.FILLED, fill_color=(0, 0, 200))
    img = cv2.addWeighted(src1=overlay, alpha=alpha, src2=img, beta=1 - alpha, gamma=0.0)
    # turn-2
    overlay = draw_rounded_rectangle(img=img, top_left=(x0-text_length*10, y0-170), bottom_right=(x0+text_length*10, y0-110), radius=10, thickness=cv2.FILLED, fill_color=(0, 0, 200))
    img = cv2.addWeighted(src1=overlay, alpha=alpha+0.2, src2=img, beta=1-alpha+0.2, gamma=0.0)

    img = draw_curve_scale(img=img, curve_measure=curve_measure)



    # place the text in the bounded box..
    img = cv2.putText(img=img, text=text, org=(x0-text_length*10+35, y0-125), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 0), thickness=2)

    cv2.imshow("Template", img)

image = np.ones((240 * 480 * 3)).reshape((240, 480, 3))

pt1 = (10, 10)
pt2 = (450, 220)


# image = cv2.imread("Demo_Design_template_Frame_of_road_detection_Status_low_level.png")
# design_image = draw_rounded_rectangle(image, pt1, pt2, radius=10, thickness=cv2.FILLED, fill_color=(0, 0, 200), alpha=0.5)
# draw_curve_scale(image.copy(), 5)


capture = cv2.VideoCapture("vid1.mp4")
frame_counter = 0
while True:
    frame_counter += 1
    if capture.get(cv2.CAP_PROP_FRAME_COUNT) == frame_counter:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        #frame_counter = 0

    _, frame = capture.read()
    print(frame.shape)

    design_template(frame, -10)

# print(image.shape)
# height, width, channels = image.shape

# print(img_dims)


# cv2.imshow("Image", image)

    if cv2.waitKey(1) == 27:
        break




"""
Upgradations:
    The text of the opencv is not as like the normal TTF fonts, so try to use TTF fonts
        {Guide: https://stackoverflow.com/questions/16615662/how-to-write-text-on-a-image-in-windows-using-python-opencv2
                https://stackoverflow.com/questions/24085996/how-i-can-load-a-font-file-with-pil-imagefont-truetype-without-specifying-the-ab}
                
        Make the rounded rectangle much more advance with the starting code of teh design_template with parameter name as "alpha_border_length"
        
                        ---------------------------- on 17th September 2020 ~ Friday.

"""