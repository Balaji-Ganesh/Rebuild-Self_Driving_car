#####
import matplotlib.pyplot as plt
print("Setting up.... ...")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
##### -- this .. when using GPU, as with GPU run, it generates lot of warnings.
import utils
from sklearn.model_selection import train_test_split
import datetime


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

# STEP-9: Train the model..
history = model.fit(utils.batchGenerator(xTrain, yTrain, batch_size=100, is_for_training=True), steps_per_epoch=300,
                    epochs=10, validation_data=utils.batchGenerator(xTest, yTest, 100, is_for_training=False), validation_steps=200)

# STEP-10: Saving the model and plotting training progress
# save the architecture and weights learned of the model
model.save('results/model_at_'+datetime.datetime.now().strftime("%Y-%m-%d_%Hhrs%Mmins")+'.h5')
print("Model saved successfully")

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylim([0, 1])
plt.legend(['Training', 'Validation'])
plt.title("Losses")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
