import utils

# STEP-1: Load the data
path = 'simulation_data'        # path of the data w.r.t. this file.
data = utils.loadTrainingSimulationData(path)


# STEP-2: Balance the data via visualization
data = utils.balanceTrainingData(data, visualize_process=False)

# STEP-3: Processing
imgPaths, steeringAngles = utils.loadData(path, data)
# print(imgPaths[0], steeringAngles[0])
