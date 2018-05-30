#!/usr/bin/env python3

# OpenCV Code to interact with Video/Image Streams go here
# Code to draw boxes returned by YOLO should also go here

# Saffat Shams Akanda, <Your Name Here>, <Your Name Here>

import cv2
import tensorflow as tf
import numpy as np
import os
import sys
import time
from darkflow.net.build import TFNet
from Person_Stats import PersonStats

# Change to false to use an Nvidia GPU with CudNN
use_cpu = True
full_yolo = True
if use_cpu:
    if not full_yolo:
        options = {"model": "cfg/tiny-yolo-voc-1c.cfg", "load": 3000, "threshold": 0.2, "gpu": 0}
    else:
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.3, "gpu": 0}
else:
    if not full_yolo:
        options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.4, "gpu": 1.0}
    else:
        options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.3, "gpu": 1.0}

box_color = [255 * np.random.rand(3)]
person_id_num = 1
current_frame = 0
frame_timer = 0
people_in_video = []
people_out_of_video = []
frameRate = []
# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


def draw_boxes(img_frame, box_info):
    # box_info in tfnet json format

    for box in box_info:
        if box["label"] == "person":
            img_frame = cv2.rectangle(img_frame, (box["topleft"]["x"], box["topleft"]["y"]),
                                      (box["bottomright"]["x"], box["bottomright"]["y"]), box_color[0], 4)

    """
    rects = []
    for box in box_info:
        if box["label"] == "person":
            rects.append([box["topleft"]["x"], box["topleft"]["y"], box["bottomright"]["x"],
                          box["bottomright"]["y"], 1])
    print(rects)
    rets = sort_alg.update(rects)
    for ret in rets:
        img_frame = cv2.rectangle(img_frame, (ret[0], ret[1]), (ret[2], ret[3]), box_color, 4)
        img_frame = cv2.putText(img_frame, ret[5], (ret[0], ret[1]), cv2.FONT_HERSHEY_COMPLEX, 2, box_color, 2)
    """
    return img_frame


# Returns the points that st says to keep, st is obtained from optical flow
# always remove index 0 because its a placeholder that breaks everything when removed from the actual array
# and this is for new updated points to draw
def pointsToTrackHelper(st, points):
    delIndexs = [0]
    for i in range(1, len(st)):
        if st[i] == 0:
            delIndexs.append(i)

    return np.delete(points, delIndexs, axis=0)


# Check if the given point is less than a threshold distance from existing points
def thresholdDistances(points, x, y):
    minDistance = 1000000000
    newPoint = np.array([x, y], dtype=np.float32)
    for point in points:
        dist = np.linalg.norm(point - newPoint)
        if dist < minDistance:
            minDistance = dist

    if minDistance > 100:
        return True
    return False


# call this at the end of each loop and it'll print out the fps every 2nd frame
def getFrameRate():
    global frame_timer
    global frameRate
    if frame_timer == 0:
        frame_timer = time.time()
        return

    frameRate.append(1.0 / (time.time() - frame_timer))
    frame_timer = 0
    print("FPS: " + str(frameRate))
    return


# Check if the point is part of the bounding box of a already existing person
def check_same_person(x, y):
    global people_in_video
    #print(people_in_video)
    for person in people_in_video:
        if not person.currentPoints:
            continue
        relative_point = person.currentPoints[0]
        x1 = relative_point[0] - person.bounding_box_size[0]
        x2 = relative_point[0] + person.bounding_box_size[1]
        y1 = relative_point[1] - person.bounding_box_size[2]
        y2 = relative_point[1] + person.bounding_box_size[3]

        if (x1 <= x <= x2) and (y1 <= y <= y2):
            return person

    return None


# Gets the bounding box distances relative to the given point
def get_bounding_box_sizes_rel(x, y, bound_box_info):
    x1 = bound_box_info["topleft"]["x"]
    x2 = bound_box_info["bottomright"]["x"]
    y1 = bound_box_info["topleft"]["y"]
    y2 = bound_box_info["bottomright"]["y"]

    return np.abs([x - x1, x2 - x, y - y1, y2 - y])


def getCenter(result):
    return (result["topleft"]["x"]+result["bottomright"]["x"])/2, (result["topleft"]["y"]+result["bottomright"]["y"])/2


def main():
    tfnet = TFNet(options)
    global current_frame
    global person_id_num
    global full_yolo
    global people_in_video
    global people_out_of_video
    if sys.argv[1] == "-i":
        print("Image File")
        # image file
        img_file = cv2.imread(sys.argv[2])
        results = tfnet.return_predict(img_file)
        cv2.imwrite("image_boxed.jpg", draw_boxes(img_file, results))

    elif sys.argv[1] == "-v":
        # video file
        print("Video File")
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        cap = cv2.VideoCapture(sys.argv[2])
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('output.avi', fourcc, 20.0, (frame_height, frame_width))
        current_frame += 1

        ret, old_frame = cap.read()
        # initialise as [[[0,0]]] this makes it work with vstack
        p0 = np.zeros((1, 2), dtype=np.float32)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                img = frame
                if len(p0) > 0:  # update points that we're tracking
                    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                    # selects good points (not sure why?)

                    # see comments for function
                    points_to_track0 = pointsToTrackHelper(st, p0)
                    points_to_track1 = pointsToTrackHelper(st, p1)

                    # update the points in the people
                    k = 0
                    person_removal_indicies = []
                    for person in people_in_video:
                        print("Checking Person", k)
                        # the person stores the indicies in p0 that belong to them, check which ones are still valid
                        indicies_to_keep = []
                        for index in person.p0_indicies:
                            if st[index] == 1:
                                indicies_to_keep.append(index)

                        print("Keeping indicies", indicies_to_keep)
                        # if they have lost any they are probably out of the frame and should not be tracked
                        if len(person.p0_indicies) < len(indicies_to_keep):
                            # person has left frame
                            print("Removing Person")
                            people_out_of_video.append(person)
                            person_removal_indicies.append(k)
                            person.last_frame_num = current_frame
                            k += 1
                            continue

                        print("Updating Points")
                        # If they are still in the frame, update the p0 indicies they contain
                        person.update_points(list(p1[indicies_to_keep]))
                        indicies_to_keep = []
                        for point in person.currentPoints:
                            i = 0
                            for new_point in points_to_track1:
                                i += 1
                                if new_point[0] == point[0] and new_point[1] == point[1]:
                                    indicies_to_keep.append(i)

                        person.update_indicies(indicies_to_keep)
                        k += 1

                        for index in person_removal_indicies:
                            del people_in_video[index]

                    # draw the tracks
                    for i, (new, old) in enumerate(zip(points_to_track1, points_to_track0)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        mask = cv2.line(mask, (a, b), (c, d), box_color[0], 2)
                        frame = cv2.circle(frame, (a, b), 5, box_color[0], -1)
                        img = cv2.add(frame, mask)

                    # normal appending and reshaping was breaking our code, so we reinitalize p0 with 0,0 (otherwise breaks)
                    # then we add on the new predicted points from optical flow
                    p0 = np.zeros((1, 2), dtype=np.float32)
                    p0 = np.vstack((p0, points_to_track1))
                    old_gray = frame_gray.copy()

                if current_frame % 10 == 0:
                    results = tfnet.return_predict(frame)
                    for result in results:
                        x, y = getCenter(result)
                        # using vstack to append new points to p0
                        if thresholdDistances(p0, x, y) and result["label"] == "person":
                            person = check_same_person(x, y)
                            if person is not None:
                                #person.add_new_point([x, y], result["confidence"], len(p0))
                                pass
                            else:
                                people_in_video.append(PersonStats(current_frame, get_bounding_box_sizes_rel(x, y,
                                                                   result), np.array([x, y], dtype=np.float32),
                                                                   result["confidence"], len(p0), person_id_num))
                                person_id_num += 1

                                p0 = np.vstack((p0, np.array([[np.float32(x), np.float32(y)]])))
            else:
                break

            cv2.imshow('frame', img)
            out.write(img)
            current_frame += 1
            getFrameRate()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print("Total People in Video: " + str(len(people_in_video) + len(people_out_of_video)))

        frame_rate = sum(frameRate) / float(len(frameRate))
        for person in people_out_of_video:
            print ("PEOPLE OUT OF VID")
            print("Person ID:", person.id)
            print("Time in Video: " + str((person.last_frame_num - person.initial_detection) / frame_rate))

        for person in people_in_video:
            print("PEOPLE STILL IN VIDEO AT END")
            print("Person ID:", person.id)
            print("Time in Video: " + str((current_frame - person.initial_detection) / frame_rate))

        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        # webcam
        print("Using Webcam")
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                results = tfnet.return_predict(frame)
                cv2.imshow("results frame", draw_boxes(frame, results))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()


