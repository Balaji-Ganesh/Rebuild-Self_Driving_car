LOG
```commandline
$ python calibrator.py captured_checkerboards png 9 6 25
32
Reading image  captured_checkerboards/capture_0.png
Pattern found! Press ESC to skip or ENTER to accept
Image accepted
Reading image  captured_checkerboards/capture_1.png
Pattern found! Press ESC to skip or ENTER to accept
Image accepted
....
Reading image  captured_checkerboards/capture_9.png
Pattern found! Press ESC to skip or ENTER to accept
Image accepted
Found 32 good images
Image to undistort:  captured_checkerboards/capture_1.png
ROI:  10 5 621 470
Calibrated picture saved as calibresult.png
Calibration Matrix: 
[[818.90158575   0.         321.99257274]
 [  0.         820.00339989 182.55204853]
 [  0.           0.           1.        ]]
Disortion:  [[-0.00414332  0.61931166 -0.02479099  0.00497604 -1.61433127]]
total error:  0.06776283246971031

```