# Labeled Cup-detection-and-tracking web service
This project makes use of detection and tracking techniques to follow a moving labeled cup from a video stream, 
it then publishes a sequence of frames with bounding rectangles around the cup and/or the label with the mention inside/outside
into a web page using a web service
## Features 
1. Detect the cup and draw a bouding box around it, moreover the mention **inside** is inserted into the frame, to acheive this result, the algorithm uses a color and contour detection technique to perform the task

2. Detect the label present in the cup and insert the mention **inside** into the frame, for this the algorithm uses the ORB feature detector to match descriptors previously extracted with the descriptors computed from the image, to make sure that the matched keypoints are indeed in the cup area, we detect the cup and double check that the keypoints is inside the cup bounding box 

## Prerequisties 
To use the code you will need Python3.x,  OpenCV and some dependencies:
1. Create and activate a new environemnt
2. Install **Flask** for creating web services using the following command:
```
pip install Flask
``` 
## Usage
Lauch script1.py from your favorite IDE or from your command prompt. In the main code you will see two variables; 
**_track_cup_label_** and **_using_tracker_**, the first one define which object to detect and track
(**0** for the **cup**, **1** for the label and **2** for both), the second variable enables the user to choose between 
using a pre-build OpenCV tracking algorithm or using detection techniques in all the frames, to define whick tracking algorthm to use, 
one can make use of the variable **_tracker_**, OpenCV offers many tracking algorithms but the most efficient one in terms of accuracy and speed is **KCF**
