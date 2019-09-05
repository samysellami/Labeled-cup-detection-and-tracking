# Labeled cup detection and tracking web service
This project makes use of detection and tracking techniques to follow a moving labeled cup in a video stream, 
it then publishes a sequence of frames with bounding rectangles around the cup and/or the label and with the mention inside/outside on the frames which are fed into web page (localhost:5000) using a web service

## Features 
1. Detect the cup and draw a bouding box around it, if the detection is successful the mention **inside** is inserted into the frame, to acheive this result, the algorithm uses a color and contour detection technique to perform the task, to track the cup, one can choose between detecting the cup in every frame or using OpenCV tracking algorithms

2. Detect the label present in the cup and insert the mention **inside** into the frame, for this,  the algorithm uses the ORB feature detector to match descriptors previously extracted (keypoints_database.p) with the descriptors computed from the image, to make sure that the matched keypoints are indeed in the cup area, we detect the cup using the steps mentioned in (1) and we double check that the keypoints are inside the cup bounding box 

## Prerequisites
To use the code you will need Python3.x,  OpenCV and some dependencies:
1. Create the environment from the object_detection_environment.yml file:
```
conda env create -f environment.yml
```

2. Activate the new environment:
- Windows: ``` activate cup_detection_env ```
- macOS and Linux: ``` source activate cup_detection_env ```

3. Verify that the new environment was installed correctly:
```
conda list
```

## Usage
Launch script1.py from your favorite IDE or from your command prompt. In the main code you will see two variables; 
**_track_cup_label_** and **_using_tracker_**, the first one define which object to detect and track
(**0** for the **cup**, **1** for the **label** and **2** for **both**), the second variable allow the user to choose between 
using a pre-build OpenCV tracking algorithm or using detection techniques in all the frames, to define whick tracking algorthm to use, you can set the variable **_tracker_** to one of the many tracking algorithms offered by OpenCV, the results shows that the most efficient in terms of accuracy and speed is **KCF**

**NB:** Using OpenCV tracking algorithms to track the objects offers poor results in comparison with the detection algorithms implemented, this is why one should consider this option for offline processing usage 
