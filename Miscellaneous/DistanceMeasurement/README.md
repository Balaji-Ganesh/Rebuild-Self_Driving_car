Goal:
----
- Measure the object's distance from camera (w.r.t. camera of course).

Principle employing
-------------------
- Object detection based on its color

Requirements
------------
- A physical object -- particularly of single color and of cuboid like shape for measurement with accuracy (physically)
- A ruler - to measure the physcical distance
- A camera

Flow
----
- Measure the physical object's height and width
- Measure the (physical) object's distance from the camera - placing straight -- !! both at same level.
- Input the measurements recorded into the program.

Results
-------
- In detection of a yellow colored TT Gear motor.
- Finding the HSV color space of the object
    - Original frame: ![Original feed](/results/Original feed_screenshot_12.03.2023.png)
    - HSV frame: ![HSV feed](/results/HSV feed_screenshot_12.03.2023.png)
    - Extracted frame: ![Extracted feed](/results/Extracted Feed_screenshot_12.03.2023.png)
- Detecting the object based on its HSV Color space
  - Mask Frame: ![](/results/Mask Frame screenshot from 2023-03-12 13-39-30.png)
  - Object : ![](/results/Object Detection screenshot 2023-03-12 13-37-03.png)
