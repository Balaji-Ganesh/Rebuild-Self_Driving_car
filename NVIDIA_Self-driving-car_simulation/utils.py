#  Importing required libraries..
import pandas as pd             # to deal with CSV files
import numpy as np              # for..
import os                       # to deal with path joining
import matplotlib.pyplot as plt  # for visualization while balancing the data
from sklearn.utils import shuffle
# ~ Why not cv2? it reads img in BGR, this in RGB. that's it.
import matplotlib.image as mpimg
from imgaug import augmenters as iaa    # for image augmentations

""" Utility functions"""


def filterFilePath(filepath):
    return filepath.split('\\')[-1]     # as needed ONLY the image name.


def loadTrainingSimulationData(path):
    # cols of the `driving_log.csv`
    cols = ['centerImg', 'leftImg', 'rightImg',
            'steeringAngle', 'throttle', 'brake', 'speed']
    # read the data..
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=cols)
    # Filter the columns required..
    data = data[['centerImg', 'steeringAngle']]
    # Truncate the image path (of its absolute path)
    data['centerImg'] = data['centerImg'].apply(filterFilePath)
    # Give an information to the user..
    print("Total images(center) loaded: "+str(data.shape[0]))
    # return the final data
    return data


def balanceTrainingData(data, visualize_process=False):
    """
        idea..
    """
    numBins = 31    # Need Odd_num as also needed `0` as center, and -ve -> left, +ve -> right (in total 3 classes)
    maxSamplesPerbin = 1000
    # to know, how much data does each class contain... using
    histogram_values, bins = np.histogram(data['steeringAngle'], numBins)
    # print("balancing data \n", histogram_values, " -- ", bins)
    """
        till here..
        `bins` - resulted values in range [-1, +1] w/o ""0"" 
        _An issue here.._
        w/o ""0"" <<- which is required, as car drives straight most of the times.
        _Employed fix_
        add it explicitly
            How it works..
            adding -> 1. all values in `bins` except last
                      2. all values in `bins` except first.
            This gives `0` in center
            An issue here..
                All the values gets doubled -- so divide by half.
    """
    if visualize_process:
        bins_with_center = (bins[:-1] + bins[1:])*0.5
        # print("bins with cener: \n", bins_with_center)
        # Visualize via bar plot.
        plt.bar(bins_with_center, histogram_values, width=0.06)
        """
            Look at the plot..
            bin - `0` has more and more values than any other -- ofcourse, drives most of the times straight. But, affects training.
            Hence trim off. -- `maxSamplesPerBin` used for this.
        """
        plt.plot((-1, 1), (maxSamplesPerbin, maxSamplesPerbin),
                 color='red')  # <<-- visualization of cutoff. !!! Check is this fine.. else adjust cutoff.
        # Now remove the redundant data of bin-0. -- so collect all indices of each tuple that exceed cutoff - to drop.
        plt.title("Plot before removal of values above cutoff")
        plt.show()
    """
        NOTE: a pre-requisite understanding, before understanding code of "removal" (below code)
        - We still havent't got the data(each tuple) that is segregated into respective bins.
        - Just got #`numBins` separation of `data` in `bins` -- i.e., only INTERVALs
        - so to remove redundant data .. 
            - take each tuple and check, in which interval it falls (take lower limit-> an interval, upper limi -> its nxt interval)
                whether it falls in that range or not.. if falls then add.
    """
    indicesOfRemovals = [
    ]          # To store the index values of tuples of redundant data (for removal)
    for binIdx in range(numBins):   # go through each bin..
        eachBinData = []            # get the values of each bin
        for i in range(len(data['steeringAngle'])):  # now in each bin..
            if data['steeringAngle'][i] >= bins[binIdx] and data['steeringAngle'][i] <= bins[binIdx+1]:
                eachBinData.append(i)               # When satisfied, add it.

        # to avoid deletion of shorter ONLY or higher ONLY (as they will be in sorted order).. Shuffle them
        eachBinData = shuffle(eachBinData)
        # removing redundant data by cutoff.
        eachBinData = eachBinData[maxSamplesPerbin:]
        indicesOfRemovals.extend(eachBinData)

    print('Images listed for removal: ', len(indicesOfRemovals))
    # drop the redundant data.. based on indices.
    data.drop(data.index[indicesOfRemovals], inplace=True)
    print('Final images (after removal): ', len(data))

    # Visualize the final result..
    if visualize_process:
        histogram_values, _ = np.histogram(data['steeringAngle'], numBins)
        plt.bar(bins_with_center, histogram_values, width=0.06)
        # <<-- visualization of cutoff. !!! Check is this fine.. else adjust cutoff.
        plt.plot((-1, 1), (maxSamplesPerbin, maxSamplesPerbin), color='red')
        # Now remove the redundant data of bin-0. -- so collect all indices of each tuple that exceed cutoff - to drop.
        plt.title("Plot after removal of values above cutoff")
        plt.show()
        """
            In the plot..
            Ensure that, data is distributed equally on either sides.
            If not, then perform training few more times.
        """
    # Now return the final Balanced and ONLY necessary data...
    return data


def loadData(path, data):
    """
    - separate the data (which is currently in pandas format) into two lists
         of image_paths(center img) and steering angle.
    """
    imgPaths = [
    ]           # list to store all the images path (with relative path of this python file)
    # list to store corresponding steering angles of each image.
    steeringAngles = []
    for idx in range(len(data)):
        # to access each tuple's value based on index. (#values in each tuple = #cols)
        indexedData = data.iloc[idx]
        # print(indexedData)
        imgPaths.append(os.path.join(
            path, 'IMG', indexedData['centerImg']))
        steeringAngles.append(float(indexedData['steeringAngle']))

    # cvt to numpy arrays
    imgPaths = np.asarray(imgPaths)
    steeringAngles = np.asarray(steeringAngles)

    # return the final values..
    return imgPaths, steeringAngles


def augmentImage(img_path, steeringAngle):
    """
    @param img_path: path of the image to be augmented.
    @param steeringAngle: respective steering angle noted. 
        Why this is needed? when flipped horizontally, directions too gets flipped. So to handle that.
    """
    img = mpimg.imread(
        img_path)        # load the image, on which augmentation has to be done.
    # Start augmentation on the loaded image.
    # PANNING - move the image left-right, up-down.. of some %. !! Does randomly
    panner = iaa.Affine(translate_percent={
                        'x': (-0.1, +0.1), 'y': (-0.1, +0.1)})
    panned_img = panner.augment_image(img)

    # ZOOMING
    zoomer = iaa.Affine(scale=(1, 1.5))
    zoomed_img = zoomer.augment_image(img)
    plt.imshow(zoomed_img)
    plt.show()


augmentImage('test.jpg', 0.5)
