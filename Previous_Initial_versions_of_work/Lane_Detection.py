# Importing the required libraries...
import cv2
import utils
import numpy as np

################################################################### GLOBAL VALUES ###########################################################
width, height = (480, 240)      # Dimensions of the window, (Currently adjusted according to the RPi camera)
window_dim = (width, height)
curvature_stack_avg_limit = 10  # What should be the length of the stack to store the elements..!! used in #STEP-4, it makes a impact on deciding the center correctly..(Please make sure that, this value is adjusted properly..else center cannot be detected properly.)
curvature_stack_list = []       # A list implemented to be like stack(FIFO), in which the elements(raw_curvature_val) are PUSHed upto the value set in "curvature_stack_avg_limit"(i.e., size of stack is this). The element which is PUSHed earlier is POPped out first(i.,e earlier raw_curvature_value is deleted, that means our stack will hold a set of previous finite values (as set in curvatur_stack_avg_limit) from current.)  Avg of this stack values is considered to be the approximated center.
first_run_setup_done = False
initial_values = [73, 107, 20, 240]   # Previously tuned warp points for video: "vid1.mp4"
#############################################################################################################################################


def get_lane_curve(image, debug_mode=False):
    """
    This function will find the curve value in the image passed and returns the curve value which is
    calculated in 5 stages:
        #STAGE-1:  Thresholding -- Separating the lane and the environment
                    (Lane as white color, environment as black color)
                    As for now, we are doing in a controlled environment, but for the real environment, its
                    in the NOTE section, please refer below.
                    (Alternative for this step is below)
        #STAGE-2: Warping -- Straightening the image acc. to the screen
                    ~~(Bird's Eye view)
        #STAGE-3: Finding the Curve value by histogram method

        #STAGE-4: Optimizing the Curve value calculated from the #STAGE-3

        #STAGE-5: Displaying the results..
                        if debug_mode=True      --> Displays the all the 4stages in each row.. along with the final result on teamplate frame..
                        if debug_mode=False     --> Only displays the final result on the template

    NOTE:
    ----
    #STAGE-1 Can be replaced by some edge detection methods like canny edge detector (core functioning explained on Quora)or
    some other detectors like explained in the video guide: https://youtu.be/LECg-Gv5xjo
        (Actually got referenced while learning the opencv in the programming knowledge channel
        at the time of making the self-driving-car project for the college)

    :param  image: The source image on which the lane should be detected.
    :param debug_mode: Whether to run in debugging mode(if True) or in normal mode(if False).
    :return: Not yet decided
    """
    """STAGE-1: Filtering the image (Manual Threshold adjustment or Automatic Edge detection)"""
    threshImg = utils.threshold_filter(image=image, debug_mode=debug_mode)                      # Filter the lane from the environment, so that we can focus only on the lane

    """STAGE-2: Warping the image(i.e., RoI) [Bird's Eye view]"""
    height, width = image.shape[:-1]                                                        # Get the width and height of the image

    # Setup that's required only for the first time.. [in multiple calls for the function, not in runs]
    # for setup of "trackbars"
    global first_run_setup_done
    if not first_run_setup_done:
        utils.initialize_warping_trackbars(initial_values=initial_values,
                                           width=window_dim[0], height=window_dim[1])                 # Initialize the Trackbars for adjusting warp area(RoI).
        first_run_setup_done = True     # to avoid execute this block in next call.

    warp_points = utils.get_warping_trackbars_values(width=window_dim[0], height=window_dim[1])   # Get the warping co-ordinates after adjustment of the warping trackbars.
    warp_img = utils.warpImage(image=threshImg.copy(), warp_points=warp_points, width=width, height=height)  # Get the warp image based on the warp_co-ordinates/

    """STAGE-3: Finding the Curve value"""
    midPoint, low_percent_hist = utils.getHistogram(warp_img.copy(), show_histogram=True, min_percent=0.5, region=4)             # Get the midpoint of the some part of the curve (i.e., some part of the bottom region)(!! Notice that region is not passed as "1" instead we passed "4")
    avg_curvature_point, histogram_img = utils.getHistogram(warp_img.copy(), show_histogram=True, min_percent=0.9, region=1)  # Get the midpoint of the whole curve that appears in the image (Notice that region's value is passed as "1")
    raw_curvature_val = avg_curvature_point - midPoint
    # print(avg_curvature_point-midPoint)

    """"STEP-4: Reducing the noise of the curve value (i.e., we take all the curve values and avg them)"""
    curvature_stack_list.append(raw_curvature_val)      # PUSH the value into the stack
    # Our stack is of finite length(as assumed to be curvature_stack_avg_limit), so POP out the first PUSHed raw_curvature_val.
    if len(curvature_stack_list) > curvature_stack_avg_limit:   # Checking whether filled or not..if yes POP out, else no
        curvature_stack_list.pop(0)                             # POPping out the first PUSHed value.

    # Take the avg of this stack..
    optimized_curve_val = int(sum(curvature_stack_list) / len(curvature_stack_list))


    """STEP-5: Displaying the Calculations performed on the images"""
    if debug_mode:  # Shows the entire work_flow..
        '''Show the flow-1: #Step_1 - Filtering the image'''
        masked_original = cv2.bitwise_and(src1=image, src2=image, mask=threshImg)  # Merge the thresholded and Original image, so that we'll get only the lane which we required.

        '''Show the flow-2: #Step_2 - Warping the image'''
        green_img = np.ones_like(image)         # Can go with other way, to get dimensions as passed image, this approach seems better..
        green_img[:] = 0, 255, 0                # Fill entire image pixels with green color
        # Merge the warpimage and the gree_background..
        masked_warp = cv2.bitwise_and(src1=cv2.cvtColor(src=warp_img, code=cv2.COLOR_GRAY2BGR), src2=green_img)

        '''Show the flow-3: #Step_3 - Finding the curve value'''
        # ## Plot the detected curve centers..##
        # First plot the midpoint..
        curve_midPt = cv2.circle(img=low_percent_hist.copy(), center=(midPoint, low_percent_hist.shape[0]-20), radius=20, color=(0, 0, 255), thickness=cv2.FILLED)
        # Next plot the avg_curvature_point..
        curve_avgPt = cv2.circle(img=histogram_img.copy(), center=(avg_curvature_point, histogram_img.shape[0]-20), radius=20, color=(255, 0, 0), thickness=cv2.FILLED)
        # Plot both the midPoint and avg_curvature_poing on a single image..
        curve_mid_avgPts = cv2.circle(img=masked_original.copy(), center=(midPoint, low_percent_hist.shape[0]-20), radius=20, color=(0, 0, 255), thickness=cv2.FILLED)
        curve_mid_avgPts = cv2.circle(curve_mid_avgPts, center=(avg_curvature_point, curve_mid_avgPts.shape[0]-20), radius=10, color=(255, 0, 0), thickness=cv2.FILLED)

        '''Show the flow-4: #Step-4 - Optimizing the Curve value (As curve value can only be printed normally, but for the display purpose we combine the "masked_img" and cv2.putText)'''
        # Get what has detected (i.e,. Road) with green color..with threshold..
        detected_road = cv2.bitwise_and(src1=cv2.cvtColor(threshImg, cv2.COLOR_GRAY2BGR), src2=green_img)
        # Add some Alpha to that with the original..
        mask_result = cv2.addWeighted(src1=image.copy(), alpha=0.8, src2=detected_road, beta=1, gamma=0)

        # Write the optimized_curve_val..
        final_result_img = cv2.putText(image.copy(), text=str(optimized_curve_val), org=(width//2-50, height//2-10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(255, 0, 0), thickness=4)
        final_result_img = cv2.circle(img=final_result_img, center=(midPoint, final_result_img.shape[0]-20), radius=20, color=(0, 255, 0), thickness=cv2.FILLED)
        final_result_img = cv2.circle(final_result_img, center=(avg_curvature_point, final_result_img.shape[0]-20), radius=10, color=(0, 255, 255), thickness=cv2.FILLED)

        # Draw a line between both the "midPoint" and "curvature_avg_val"
        cv2.line(final_result_img, pt1=(midPoint, final_result_img.shape[0]-20), pt2=(avg_curvature_point, final_result_img.shape[0]-20), color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

        # print(width//2, height//2-30)
        # curve_mid_avgPts = cv2.circle(curve_mid_avgPts, center=(avg_curvature_point, curve_mid_avgPts.shape[0]), radius=10, color=(255, 0, 0), thickness=cv2.FILLED)


        # Concat all the images..
        concat_img = utils.display_concatenated_results(images=[[image.copy(), threshImg, masked_original],         # Flow of #STEP-1
                                                                [warp_img, histogram_img, masked_warp],             # Flow of #STEP-2
                                                                [curve_midPt, curve_avgPt, curve_mid_avgPts],       # Flow of #STEP_3
                                                                [detected_road, mask_result, final_result_img]],    # # Just writing the optimized value from #STEP-3 (with some extra decoration..)
                                                        image_names=[["Original feed", "Thresholded image", "Thresh + Original"],
                                                                     ["Warped Image", "Histogram  ", "Masked Warp  "],
                                                                     ["Curve Mid point", "Curvature Avg Center", "Curve's Mid & Avg Pts"],
                                                                     ["Detected Road (on Thresh)", "Detected Road (Original)", "Optimized(Avg) Steering Angle"]],
                                                        scaleFactor=0.7)
        cv2.imshow('Workflow {Concatenated images} "Each row denotes Each STEP"', concat_img)

    # shows the result on the template frame
    utils.design_template(image, optimized_curve_val)


def start_detecting_lanes(debug_mode=False):
    """
    This function works like a supervisor to implement a task of detect the lanes.


    :return: nothing.
    """

    capture = cv2.VideoCapture("road_car_view.mp4")      # Get the camera capture instance..
    frame_counter = 0
    while True:
        frame_counter += 1
        if capture.get(cv2.CAP_PROP_FRAME_COUNT) == frame_counter:  # If reached to the last frame(i.e., to end of the video..)
            capture.set(cv2.CAP_PROP_POS_FRAMES, 0)                # Set the frame-count to 0, so we start again from start..
            frame_counter = 0                                   # set to 0, as now the current frame is 0th frame

        read_status, frame = capture.read()         # Get the feed from the capture instance..
        if (read_status == True):
            frame = cv2.resize(frame, window_dim)   # Resize the frame size acc. to the RPi camera size (its of 480 X 240)
            cv2.imshow("Video", frame)              # Display the image..
        else:
            print ("Cannot get the frame..")

        # Find the Curve of the road..
        get_lane_curve(image=frame, debug_mode=debug_mode)

        if cv2.waitKey(1) == 27:
            if debug_mode:
                warp_points = utils.get_warping_trackbars_values()
                print("Warping values (width_top, height_top), (width_bottom, height_bottom): "+str(warp_points[0::2]))
            break


if __name__ == '__main__':
    start_detecting_lanes(debug_mode=False)