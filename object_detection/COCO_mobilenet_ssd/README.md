- [COCO model](https://cocodataset.org)
- obtained from
    - weights file - [`frozen_inference_graph.pb`](https://github.com/zafarRehan/object_detection_COCO/blob/main/frozen_inference_graph.pb)
    - [`ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt`](https://github.com/zafarRehan/object_detection_COCO/blob/main/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt)
    - [`labels.txt`](https://github.com/zafarRehan/object_detection_COCO/blob/main/labels.txt) -- !! it contains only `80` names, but with this file, some times, will get error as `.. index out of range` -- that's because, it doesn't contain all class names. This file in repo, is updated with 91 class names.
- Tutorial followed [Object Detection OpenCV Python | Easy and Fast (2020) - Murtaza Hassan](https://youtu.be/HXDD7-EnGBY)

### Final report on 9th March, 2023 - Thu
- This model was tried for the purpose - whether this **can detect traffic lights or not**.
    - There is traffic light class [class_10], but it isn't detecting.
    - May be need to have custom trained model.