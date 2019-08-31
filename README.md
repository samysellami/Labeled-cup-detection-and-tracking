# Labeled Cup-detection-and-tracking web service
This project makes use of detection and tracking techniques to follow a moving labeled cup from a video stream, 
it then publishes a sequence of frames with bounding rectangles around the cup and/or the label with the mention inside/outside
into a web page using a web service
## Features 
Detect the cup and draw a bouding box around it, when the cup is detected the mention **inside** is inserted into the frame

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
