import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny(img):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Reduce the color space
    # Smooth out the sharp edges -- so as to avoid detect all erroneous edges
    kernel = (5, 5)
    blur_img = cv2.GaussianBlur(gray_img, kernel, 0)
    # Now, detect the edges.
    canny_img = cv2.Canny(blur_img, 50, 150) ################################ !!!!!!!!!!!!!!!!!!!!!! Playable values
    return canny_img

def region_of_interest(img):
    height = img.shape[0]
    width = img.shape[1]
    # Create a mask like the clone of of `img`.
    mask = np.zeros_like(img)
    # Define the triangle points -- the area of interst.
    triangle = np.array([[(2, 212), (126, 20), (316, 216)]], np.int32)
    # Now fill the above created `mask` with white in the `triangle` region
    cv2.fillPoly(mask, triangle, 255)
    # Now get only the region of interest -- from the image.
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def get_houghLines(img):
    """
    motivation: as humans, by seeing the road lanes image, can tell, where they are going towards. 
    But, the same for a computer -- just a set of pixels.
    This funciton helps in resolving that.
    Based on analysing the pattern of the pixels, approximates the direction, to where they are going.
    """
    houghLines = cv2.HoughLinesP(image=img, rho=2, theta=np.pi/180, threshold=100, lines=np.array([]), minLineLength=40, maxLineGap=5)
    return houghLines


def draw_houghLines(img, houghlines):
#     drawn_image = np.zeros_like(img)
    if houghlines is not None:
        for line in houghlines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 10)
    return img


def get_lane_coordinates(img, line):
    """
    @param img:
    @param line (slope, intercept):
    
    Notes
    Look the resource of TDS blog.
    """
    slope, intercept = line
    height, width = img.shape
    ## Assumptions taking..
    y1 = int(height) # y1 at the bottom
    y2 = int(y1*3.0/5)  # y2 -- some where between top and bottom
    # now from `y=mx+c`, x1 and x2 are calculated.
    x1 = int((y1-intercept)/slope)
    x2 = int((y2-intercept)/slope)
    
    # done.
    return [[x1, y1, x2, y2]]

def get_avg_slope_intercepts(img, lines):
    """
    @brief takes the `lines` (houghlines) - classifies into left_lane and right_line.
    then performs average on those.
    Which yields in single line at both sides.
    
    Notes
    - np.polyfit() takes the co-ordinates -- analyzes and gives certain parameters describing it.
        in this case, its a line(deg=1), those parameters will be **slope** and **intercept**
    - if the slope is -ve - classify as left_lane,
                      +ve               right_lane
        figure out how... RayanSlim - PgmngKnowledge - did work on it.
    """
    ## classify the set of given lines into two classes
    left_fit = []
    right_fit = []
#     print(lines)
    for line in lines:
#         print(line[0])
        x1, y1, x2, y2 = line[0]
        slope, intercept = np.polyfit(x=(x1, x2), y=(y1, y2), deg=1)
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))
    
    # Now, find the average..
    left_fit_avg = np.average(left_fit, axis=0)
    right_fit_avg = np.average(right_fit, axis=0)
    
    # Now had averages of `slope` and `intercept` of either sides.
    # - but, we can't draw line based on `m` and `c`.
    # - so, get the co-ordinates based on these.
    left_lane = get_lane_coordinates(img, left_fit_avg)
    right_lane = get_lane_coordinates(img, right_fit_avg)
    
    # done..!! return the results
    return [left_lane, right_lane]

def draw_final_lanes(img, lanes):
    if lanes is not None:
        for lane in lanes:
            print(lane)
            x1, y1, x2, y2 = lane[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 10)
    return img

def main():
    capture = cv2.VideoCapture('test1.mp4')
    
    while True:
        ret, frame = capture.read()
        # Apply canny edge detection
        img = canny(frame)
        # Remove the clutter
        roi_img = region_of_interest(img)
        # Get the hough lines
        houghlines = get_houghLines(roi_img)
        # Draw the hough lines on the image
        # houglined_img = draw_houghLines(frame, houghlines)
        # plt.imshow(houglined_img)
        # Get the single left and right lane co-ordinates
        road_lanes = get_avg_slope_intercepts(img, houghlines)
        # Draw the final lanes.
        img = draw_final_lanes(frame, road_lanes)
        cv2.imshow("Lane Detector", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()