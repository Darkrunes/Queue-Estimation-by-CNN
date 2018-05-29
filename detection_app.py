#!/usr/bin/env python3

# OpenCV Code to interact with Video/Image Streams go here
# Code to draw boxes returned by YOLO should also go here

# Saffat Shams Akanda, <Your Name Here>, <Your Name Here>

import cv2
import tensorflow as tf
import numpy as np
import os
import sys
from darkflow.net.build import TFNet

# Change to false to use an Nvidia GPU with CudNN
use_cpu = True
if use_cpu:
    options = {"model": "cfg/tiny-yolo-voc-1c.cfg", "load": 3000, "threshold": 0.2, "gpu": 0}
else:
    options = {"model": "cfg/tiny-yolo-voc.cfg", "load": "bin/tiny-yolo-voc.weights", "threshold": 0.4, "gpu": 1.0}

box_color = [255 * np.random.rand(3)]

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


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

def getCenter(result):
    return (result["topleft"]["x"]+result["bottomright"]["x"])/2, (result["topleft"]["y"]+result["bottomright"]["y"])/2


def main():
    tfnet = TFNet(options)
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

        ret, old_frame = cap.read()
        p0 = np.array([[[0.0, 0.0]]], dtype=np.float32)
        # np.delete(p0, 0)
        old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(old_frame)

        while cap.isOpened():
            ret, frame = cap.read()
            print (p0)
            img = frame
            if (len(p0) > 0 ):# update points that we're tracking
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
                print ("done")
                # selects good points (not sure why?)
                points_to_track0 = p0[st == 1]
                points_to_track1 = p1[st == 1]

                # draw the tracks
                for i,(new,old) in enumerate(zip(points_to_track1,points_to_track0)):
                    a,b = new.ravel()
                    c,d = old.ravel()
                    mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
                    frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
                    img = cv2.add(frame,mask)
                

                p0 = points_to_track1.reshape(-1,1,2)
                old_gray = frame_gray.copy()


            if ret: # find any new feature points (center of boxes)
                results = tfnet.return_predict(frame)
                for result in results:
                    x, y = getCenter(result)
                    p0 = np.append(p0, [[[x,y]]], axis=0)
                    # add center points to feature point if no feature points are inside the box
            cv2.imshow('frame',img)
            # set up previous values for next iteration
                        # update good features using optical flow

            # 
            # if ret:
            #     results = tfnet.return_predict(frame)
            #     # tracker defintion
            #     boxed_image = draw_boxes(frame, results)
            #     out.write(boxed_image)
            #     cv2.imshow("results frame", boxed_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    else:
        #webcam
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


