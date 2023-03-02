import utils

# STEP-1: Load the data
path = 'simulation_data'
data = utils.loadTrainingSimulationData(path)


# STEP-2: Balance the data via visualization
data = utils.balanceTrainingData(data, visualize_process=False)
