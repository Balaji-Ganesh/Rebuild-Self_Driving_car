# Import the libraries required.
import os   # for dealing with filepaths,,
import cv2
import time # for putting timestamp on image names

## Initial setup..
output_path = 'data/images'            # Path, where the images are to be saved
camera_brightness = 100     # brightness of camera
i_th_frame  = 10            # Consider every i'th frame
min_blur = 500              # blur value, below which should not be accepted
save_as_gray_scale = False   # whether to save as grayscale or colored
save_data = True            # Whether in testing mode or in actual mode
display_process = True      # Whether to show preview of capture or not
img_width = 100             # width of img
img_height = 120            # height of img

images_saved_till_now = 0
frames_count = 0
folder_count = 0

def createFoldersToSaveImages():
    global folder_count
    # Get existing folder count in the `output_path`
    while os.path.exists(output_path + '_' + str(folder_count)):
        folder_count += 1
    os.makedirs(output_path +'_'+ str(folder_count))

if __name__ == '__main__':
    capture = cv2.VideoCapture(0)

    if save_data:
        createFoldersToSaveImages()
    blurness = 0
    while True:
        _, frame = capture.read()     # Read the frame
        frame = cv2.resize(frame, (img_width, img_height))    # also try imgutils
        if save_as_gray_scale: frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
        if save_data:
            blurness = cv2.Laplacian(frame, cv2.CV_64F).var()   # calculate the blurness of the image captured
            if frames_count % i_th_frame == 0 and blurness >= min_blur:     # Apply the filters..
                current_time = time.time()
                cv2.imwrite(output_path  + '_' + str(folder_count) + '/' + str(images_saved_till_now)+'_'+str(int(blurness)) + "_"+str(current_time)+'.png', frame)
                images_saved_till_now += 1
            frames_count += 1

        if display_process:
            cv2.putText(img=frame, text=str(blurness), org=(10, 10), fontFace=cv2.FONT_HERSHEY_COMPLEX, color=(0,0,0), fontScale=0.5, thickness=1)
            cv2.imshow("Capture data", frame)
        
        if cv2.waitKey(1) &0xFF == ord('q'):
            break
    
    capture.release()
    cv2.destroyAllWindows()

print("------------------ Stats ------------------")
print("Total frames captured            : ", frames_count)
print("Total frames considered and saved: ", images_saved_till_now)
print("additional info: \"meaning of image_name\": <count>_<blurness_value>_<time_stamp>.png")