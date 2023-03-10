## Purpose..
- :idea: 
    - If any object detection mechanism worked on, doesn't suit the needs of RPi computation, this works like a backup -- as traffic signs used are of different shape (and area), so can use that.
    - also planning for.. if the distance measurement doesn't work properly, by this, area of traffic lights can be calculated ... as.
        - as the car moves near to the traffic light, its area on frame increases
    
: Update after experiment..
- it appears, that, this is very susceptible to the light conditions and need tweaking of parameters based on light conditions -- which is infeasible at real-time.

: One more update..
    - This time, traffic lights are being detected.
    - The catch is.. detecting, when any one of the light is glowing. 
        Previously trying on traffic lights (all lights OFF state).
        Guess, they haven't trained over traffic lights in off state.
    