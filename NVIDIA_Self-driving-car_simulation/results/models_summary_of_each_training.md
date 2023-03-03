# model summary on 3rd March, 2023 - Friday

```
[arjuna@kurukshetra NVIDIA_Self-driving-car_simulation]$ python learningFromSimulation.py 
Setting up.... ...
Total images(center) loaded: 16357
Images listed for removal:  11891
Final images (after removal):  4466
Training set size:  3572
Test set size:  894
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv2d (Conv2D)             (None, 31, 98, 24)        1824      
                                                                 
 conv2d_1 (Conv2D)           (None, 14, 47, 36)        21636     
                                                                 
 conv2d_2 (Conv2D)           (None, 5, 22, 48)         43248     
                                                                 
 conv2d_3 (Conv2D)           (None, 3, 20, 64)         27712     
                                                                 
 conv2d_4 (Conv2D)           (None, 1, 18, 64)         36928     
                                                                 
 flatten (Flatten)           (None, 1152)              0         
                                                                 
 dense (Dense)               (None, 100)               115300    
                                                                 
 dense_1 (Dense)             (None, 50)                5050      
                                                                 
 dense_2 (Dense)             (None, 10)                510       
                                                                 
 dense_3 (Dense)             (None, 1)                 11        
                                                                 
=================================================================
Total params: 252,219
Trainable params: 252,219
Non-trainable params: 0
_________________________________________________________________
```
## Training
with `steps_per_epoch`=#batches=20 and 2 epochs
results:
```
Epoch 1/2
20/20 [==============================] - 12s 521ms/step - loss: 0.1813 - val_loss: 0.1843
Epoch 2/2
20/20 [==============================] - 9s 458ms/step - loss: 0.1359 - val_loss: 0.1573
```
- after some time -- same configuration
```
Epoch 1/2
20/20 [==============================] - 8s 380ms/step - loss: 0.1714 - val_loss: 0.1291
Epoch 2/2
20/20 [==============================] - 6s 325ms/step - loss: 0.1449 - val_loss: 0.1251
Model saved successfully
```
and its loss plot..
![Loss plot](loss_plot_of_training_at_2023-03-03_14hrs19mins.png)

### Actual parameters to train on
```python
model.fit(utils.batchGenerator(xTrain, yTrain, batch_size=100, 
          is_for_training=True), steps_per_epoch=300,
          epochs=10, validation_data=utils.batchGenerator(xTest, yTest, 100,
          is_for_training=False), validation_steps=200)

```
- on `100*300 = 30,000` images for training and `100*200=20,000` for testing
    ```sh
    [arjuna@kurukshetra NVIDIA_Self-driving-car_simulation]$ time python learningFromSimulation.py 
    Setting up.... ...
    Total images(center) loaded: 16357
    Images listed for removal:  11891
    Final images (after removal):  4466
    Training set size:  3572
    Test set size:  894
    Model: "sequential"
    _________________________________________________________________
    Layer (type)                Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)             (None, 31, 98, 24)        1824      
                                                                    
    conv2d_1 (Conv2D)           (None, 14, 47, 36)        21636     
                                                                    
    conv2d_2 (Conv2D)           (None, 5, 22, 48)         43248     
                                                                    
    conv2d_3 (Conv2D)           (None, 3, 20, 64)         27712     
                                                                    
    conv2d_4 (Conv2D)           (None, 1, 18, 64)         36928     
                                                                    
    flatten (Flatten)           (None, 1152)              0         
                                                                    
    dense (Dense)               (None, 100)               115300    
                                                                    
    dense_1 (Dense)             (None, 50)                5050      
                                                                    
    dense_2 (Dense)             (None, 10)                510       
                                                                    
    dense_3 (Dense)             (None, 1)                 11        
                                                                    
    =================================================================
    Total params: 252,219
    Trainable params: 252,219
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 1/10
    300/300 [==============================] - 272s 905ms/step - loss: 0.1252 - val_loss: 0.0983
    Epoch 2/10
    300/300 [==============================] - 183s 610ms/step - loss: 0.1055 - val_loss: 0.0963
    Epoch 3/10
    300/300 [==============================] - 183s 610ms/step - loss: 0.1019 - val_loss: 0.0833
    Epoch 4/10
    300/300 [==============================] - 178s 593ms/step - loss: 0.0943 - val_loss: 0.0794
    Epoch 5/10
    300/300 [==============================] - 159s 532ms/step - loss: 0.0899 - val_loss: 0.0767
    Epoch 6/10
    300/300 [==============================] - 173s 578ms/step - loss: 0.0847 - val_loss: 0.0744
    Epoch 7/10
    300/300 [==============================] - 200s 666ms/step - loss: 0.0839 - val_loss: 0.0701
    Epoch 8/10
    300/300 [==============================] - 191s 636ms/step - loss: 0.0821 - val_loss: 0.0712
    Epoch 9/10
    300/300 [==============================] - 203s 678ms/step - loss: 0.0798 - val_loss: 0.0705
    Epoch 10/10
    300/300 [==============================] - 188s 626ms/step - loss: 0.0794 - val_loss: 0.0675
    Model saved successfully

    real    33m5.157s
    user    75m46.540s
    sys     14m28.397s
    ```
    - loss plot..
    ![loss plot on actual parameters](loss_plot_of_training_at_2023-03-03_15hrs08mins.png)
    - !!!! This plot is not similar to the guide (but the architecture was exact)