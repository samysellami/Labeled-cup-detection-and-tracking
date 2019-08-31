
from flask import Flask, request, render_template, send_from_directory
import os
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = (10, 8)  # (w, h)
import numpy as np
import sys
import cv2
import pickle

app = Flask(__name__)

@app.route('/')
def get_gallery():
    image_names = os.listdir('./static')
    image_names = sorted(image_names, key=lambda x: int(x.split('_')[1].split('.')[0]))
    return render_template("gallery.html", image_names=image_names)


# -*- coding: utf-8 -*-
"""Test_EdgeVision

"""

"""#Methods used
##Extracting the cup and computing the descriptors
"""
def get_keypoints(frame):
    # Initiate ORB detector
    orb = cv2.ORB_create(100000)
    img_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    # compute the descriptors with ORB
    kp1 = orb.detect(img_gray, None)
    # compute the descriptors with ORB
    kp1, des1 = orb.compute(img_gray, kp1)
    return kp1, des1

def pickle_keypoints(keypoints):
    ###### function to save the keypoints
    i = 0
    temp_array = []
    for point in keypoints:
        temp = (point.pt, point.size, point.angle, point.response, point.octave, point.class_id)
        ++i
        temp_array.append(temp)
    return temp_array

def unpickle_keypoints(array):
    ##### function to load the keypoints
    keypoints = []
    for point in array:
        temp_feature = cv2.KeyPoint(x=point[0][0], y=point[0][1], _size=point[1], _angle=point[2], _response=point[3],
                                    _octave=point[4], _class_id=point[5])
        keypoints.append(temp_feature)
    return keypoints

"""## Thresholding"""

def thresholding(frame, thresh_val, use_otsu, block_size, use_morph_ope, num_morph, show_plot):
    ######## thersholding the image
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.GaussianBlur(img_gray, (block_size, block_size), 0)
    img_gray = cv2.medianBlur(img_gray, block_size)

    # adaptive theresholding
    if use_otsu:
        ret, img_bw = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        ret, img_bw = cv2.threshold(img_gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
    kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=(block_size, block_size))

    # morphological operations
    if use_morph_ope:
        for i in range(num_morph):
            img_bw = cv2.morphologyEx(img_bw, cv2.MORPH_OPEN, kernel)
            img_bw = cv2.morphologyEx(img_bw, cv2.MORPH_CLOSE, kernel)
    if show_plot:
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(121), plt.imshow(frame), plt.title('input image')
        plt.subplot(122), plt.imshow(img_bw, 'gray'), plt.title('image threshold'), plt.show()
    return img_bw

"""## Contour detection:"""

def detect_color(frame, show_plot):
    # convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # define the HSV range of the blue color
    lower_blue = np.array([100, 0, 0], np.uint8)
    upper_blue = np.array([160, 255, 255], np.uint8)

    # threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

    # bitwise-AND mask and original image
    img_color = cv2.bitwise_and(frame, frame, mask=mask)
    if show_plot:
        fig = plt.figure(figsize=(10, 6))
        plt.subplot(131), plt.imshow(frame), plt.title('Input image')
        plt.subplot(132), plt.imshow(mask), plt.title('mask')
        plt.subplot(133), plt.imshow(img_color), plt.title("Color image"), plt.show()
    return img_color


def detect_contours(frame, show_plot):
    # threshold the image
    use_otsu = 1;
    block_size = 5;
    use_morph_ope = 1;
    num_morph = 1;
    thresh_val = 11
    img_bw = thresholding(frame, thresh_val, use_otsu, block_size, use_morph_ope, num_morph, show_plot)
    # find contours
    im2, contours, hierarchy = cv2.findContours(img_bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area = frame.shape[0] * frame.shape[1]
    # approximate the contours detected
    approx_list = [];
    for i in range(len(contours)):
        cnt = contours[i]
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        if (cv2.contourArea(approx) < 0.95 * area and cv2.contourArea(approx) > 0.01 * area):
            approx_list.append(approx)

    if len(approx_list) == 0:
        cnt_cup = []
    else:
        cnt_cup = approx_list[0]
        for approx in approx_list:
            if cv2.contourArea(approx) > cv2.contourArea(cnt_cup):
                cnt_cup = approx
        if show_plot:
            img_contours = frame.copy()
            cv2.drawContours(img_contours, [cnt_cup], -1, (0, 255, 0), 3)
            fig = plt.figure(figsize=(10, 6))
            plt.subplot(121), plt.imshow(frame), plt.title('original image')
            plt.subplot(122), plt.imshow(img_contours), plt.title('contours image'), plt.show
        cnt_cup = cnt_cup[:, 0, :]
    return cnt_cup

"""#Tracking

## Detections methods
"""

def show_frame(img, boxes, inside, texts, title, show_plot, colors):
    ####### Draw objects rectangles on the frame
    if inside:
        for i, box in enumerate(boxes):
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(img, p1, p2, colors[i], 3)
            cv2.putText(img, texts[i], p2, cv2.FONT_HERSHEY_SIMPLEX, 2, colors[i], 3, cv2.LINE_AA)

        cv2.rectangle(img, (10, 20), (250, 90), (255, 255, 255), -1)
        cv2.putText(img, 'inside', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (10, 20), (250, 90), (255, 255, 255), -1)
        cv2.putText(img, 'outside', (15, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4, cv2.LINE_AA)
        # enumerate the frames
    cv2.rectangle(img, (10, 100), (145, 160), (255, 255, 255), -1)
    cv2.putText(img, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3,
                cv2.LINE_AA)

    if show_plot:
        fig = plt.figure(figsize=(6, 6))
        plt.imshow(img), plt.title(title), plt.show()


def color_detect_object(frame, show_plot):
    ##### function to detect the cup using the color
    show = 0
    # detect the color blue in the image
    img_color = detect_color(frame, show)
    # find the contours of the result image
    cnt_cup = detect_contours(img_color, show)
    if len(cnt_cup) != 0:
        inside = 1
        # define the boundaries of the rectangle
        x_points = np.sort(cnt_cup[:, 0])
        y_points = np.sort(cnt_cup[:, 1])
        x_points = x_points[x_points > 5];
        x_points = x_points[x_points < frame.shape[1] - 5]
        y_points = y_points[y_points > 5];
        y_points = y_points[y_points < frame.shape[0] - 5]

        left = x_points[0];
        right = x_points[-1]
        top = y_points[0];
        bottom = y_points[-1]
        box = (left, top, (right - left), (bottom - top))
    else:
        inside = 0
        box = (0, 0, 0, 0)
    # show the frame with the bounding rectangles
    if show_plot:
        show_frame(frame, [box], inside, ['cup'], 'detection', show_plot, [(255, 255, 255)])
    return frame, box, inside


def ORB_detect_object(frame, kp1, des1, min_dist_match, show_plot):
    ##### function to detect the label using ORB feature descriptor
    # extract descriptors
    kp2, des2 = get_keypoints(frame)
    # match keypoints
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # Sort them in the order of their distance.
    matches = sorted(matches, key=lambda x: x.distance)
    distances = [match.distance for match in matches]
    distances = np.array(distances)
    #   print('the least matches distance is {}'.format(distances[0]))
    #   matches  = [match for match in matches if match.distance < 5*np.std(distances)]

    if len(matches) > 0 and matches[0].distance < min_dist_match:
        matches = [matches[0]]
        keypoints = []
        # compute the position of the center of the keypoints
        for match in matches:
            keypoints.append(kp2[match.trainIdx])
        centers_x = [keypoint.pt[0] for keypoint in keypoints];
        centers_x = np.array(centers_x);
        center_x = np.mean(centers_x)
        centers_y = [keypoint.pt[1] for keypoint in keypoints];
        centers_y = np.array(centers_y);
        center_y = np.mean(centers_y)
        center = (center_x, center_y)

        width = 50;
        left = int(center_x - width / 2)
        height = 20;
        top = int(center_y - height / 2)
        box = (left, top, width, height)
        inside = 1
    else:
        center = (0, 0)
        inside = 0
        box = (0, 0, 0, 0)
    # show the frame with the bounding rectangles
    if show_plot:
        show_frame(frame, [box], inside, ['label'], 'detection', show_plot, [(0, 255, 0)])
    return frame, box, center, inside


"""## Tracking"""

def createTrackerByName(trackerType):
    # define the trackers types
    trackerTypes = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']

    # Create a tracker based on tracker name
    if trackerType == trackerTypes[0]:
        tracker = cv2.TrackerBoosting_create()
    elif trackerType == trackerTypes[1]:
        tracker = cv2.TrackerMIL_create()
    elif trackerType == trackerTypes[2]:
        tracker = cv2.TrackerKCF_create()
    elif trackerType == trackerTypes[3]:
        tracker = cv2.TrackerTLD_create()
    elif trackerType == trackerTypes[4]:
        tracker = cv2.TrackerMedianFlow_create()
    elif trackerType == trackerTypes[5]:
        tracker = cv2.TrackerGOTURN_create()
    elif trackerType == trackerTypes[6]:
        tracker = cv2.TrackerMOSSE_create()
    elif trackerType == trackerTypes[7]:
        tracker = cv2.TrackerCSRT_create()
    else:
        tracker = None
        print('Incorrect tracker name')
    return tracker


def track_object(frame, video, bboxes, tracker, track_cup_label):
    # define the parameters of the object to track
    if track_cup_label == 1:
        texts = ['label', 'cup']
        colors = [(0, 255, 0), (255, 255, 255)]
    elif track_cup_label == 1:
        texts = ['cup', 'label']
        colors = [(255, 255, 255), (0, 255, 0)]
    else:
        texts = ['cup', 'label']
        colors = [(255, 255, 255), (0, 255, 0)]

    # initialize OpenCV's special multi-object tracker
    multiTracker = cv2.MultiTracker_create()
    # Initialize MultiTracker
    for bbox in bboxes:
        multiTracker.add(createTrackerByName(tracker), frame, bbox)

    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print(" Cannot read the video file or End of the video stream !! ")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # get updated location of objects in subsequent frames
        success, boxes = multiTracker.update(frame)
        if success:
            # objects detected
            inside = 1
        else:
            # no objects detected
            inside = 0
            break

        # show the frame with the bounding rectangles
        show_frame(frame, boxes, inside, texts, 'tracking', 1, colors)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('static/photo_{}.png'.format(capture.get(cv2.CAP_PROP_POS_FRAMES)), frame)

    return frame

"""## Detection"""

def detect_object(frame, capture, kp1, des1, track_cup_label, using_tracker):
    #### function to detect the object using colors and keypoints
    while True:
        min_dist_match = 35
        if track_cup_label == 0:  # track the cup
            img_cup, box2, inside = color_detect_object(frame, 0)
            boxes = [box2]
            text = ['cup']
            colors = [(255, 255, 255)]

        elif track_cup_label == 1:  # track the label
            img_cup, box2, inside = color_detect_object(frame, 0)
            img_ORB, box1, center, inside = ORB_detect_object(frame, kp1, des1, min_dist_match, 0)
            # if the label detected is not inside the cup detected discard the result
            if center[0] < box2[0] or center[0] > box2[0] + box2[2] or center[1] < box2[1] or center[1] > box2[1] + \
                    box2[3]:
                inside = 0
                box1 = (0, 0, 0, 0)
            boxes = [box1]
            text = ['label']
            colors = [(0, 255, 0)]

        else:  # track both the cup and the label
            img_ORB, box1, center, inside = ORB_detect_object(frame, kp1, des1, min_dist_match, 0)
            img_cup, box2, inside = color_detect_object(frame, 0)
            text = ['cup', 'label']
            colors = [(255, 255, 255), (0, 255, 0)]
            if center[0] < box2[0] or center[0] > box2[0] + box2[2] or center[1] < box2[1] or center[1] > box2[1] + \
                    box2[3]:
                if using_tracker:
                    inside = 0
                    box2 = (0, 0, 0, 0)
                boxes = [box2]
            else:
                boxes = [box2, box1]

        # show the frame with bounding rectangles
        show_frame(frame, boxes, inside, text, 'detection', 1, colors)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite('static/photo_{}.png'.format(capture.get(cv2.CAP_PROP_POS_FRAMES)), frame)
        if capture.get(cv2.CAP_PROP_POS_FRAMES) ==5:
            frame = None
            break

        if using_tracker and inside:
            return frame, boxes
        # read new frame
        ret, frame = capture.read()
        if frame is None:
            print(" Cannot read video file or  End of the video stream !! ")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame, boxes


"""# main code"""

if __name__ == "__main__":

    capture = cv2.VideoCapture('Test_video.ogv')

    if not capture.isOpened:
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = capture.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Retrieve Keypoint Features
    keypoints_database, des1 = pickle.load(open("keypoints_database.p", "rb"))
    kp1 = unpickle_keypoints(keypoints_database[0])

    # set track_cup_label to 0 to track the cup, 1 to track the label and 2 to track both
    track_cup_label = 0
    # use tracking algorithm to track the object
    using_tracker = 0
    # define the tracking algorithm
    tracker = "KCF"

    while True:
        frame, bbox = detect_object(frame, capture, kp1, des1, track_cup_label, using_tracker)
        if frame is None:
            break
        frame = track_object(frame, capture, bbox, tracker, track_cup_label)
        if frame is None:
            break

    # run the web service
    app.run()

