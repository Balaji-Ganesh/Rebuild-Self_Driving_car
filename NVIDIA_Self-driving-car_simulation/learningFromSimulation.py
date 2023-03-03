#####
print("Setting up.... ...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##### -- this .. when using GPU, as with GPU run, it generates lot of warnings.
import utils
from sklearn.model_selection import train_test_split


# STEP-1: Load the data
path = 'simulation_data'        # path of the data w.r.t. this file.
data = utils.loadTrainingSimulationData(path)


# STEP-2: Balance the data via visualization
data = utils.balanceTrainingData(data, visualize_process=False)

# STEP-3: Processing
imgPaths, steeringAngles = utils.loadData(path, data)
# print(imgPaths[0], steeringAngles[0])

# STEP-4: Da ta splitting(into training and validation)
xTrain, xTest, yTrain, yTest = train_test_split(
    imgPaths, steeringAngles, test_size=0.2, random_state=5)
print("Training set size: ", len(xTrain))
print("Test set size: ", len(xTest))

# STEP-5: Data Augmentation - helps in generalizing the model
## Need: Even we have lots of data, it falls shortage
## with these simple techniques, we can create more data from the data we had.

# STEP-6: Pre-processing
## Neeed:

# STEP-7: Batch Generator

# STEP-8: Model creation
# Going to use the same model used and tested by NVIDIA.
model = utils.modelCreator()
model.summary()
