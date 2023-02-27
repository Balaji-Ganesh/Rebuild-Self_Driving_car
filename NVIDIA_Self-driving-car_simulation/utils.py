#  Importing required libraries..
import pandas as pd     # to deal with CSV files
import numpy as np      # for..
import os               # to deal with path joining

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


"""
Make the images path as "relative"
    create a function
    use `apply`
    make a check.cc 
"""
