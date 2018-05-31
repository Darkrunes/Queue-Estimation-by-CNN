Goals:
To Detect the number of people in the video
To track the wait time of each detected person in the video
To track the average wait time
To maintain decent FPS
------------

The darkflow code and cfg files are obtained through : https://github.com/thtrieu/darkflow
The repository has instructions on how to set that up
------------

Setup 

Install all the python3 libraries listed in requirements.txt
Make sure you have gnu-gcc (Linux or MacOS) or Visual C++ 14 (Windows)

The weight files for our Tiny-YOLO and YOLOv2 (fine tuned networks) have not been provided
as per Manna's comment on the report page. 

------------
Scripts/Programs

Our main file is detection_app.py (Python3)
Usage python3 detection_app.py 
-i "image_name"		To get our bounding box detection for an image		(Deprecated)
-v "video_name"		Runs the given video showing the current detections, and tracked points
When run in this way the code can be stopped early by pressing "q".
When it runs through the whole video or stopped early; it will 
for each person show their:
	Label ID NUM
	Weather or not they were in the frame at the end of the video
	How long they were in the video
The average wait time over all detected people
The average calculated FPS

	Uses Person_Stats.py to help with tracking

To run the file with GPU change line 18 "use_cpu" to False
To run the file with Tiny YOLOv2 instead of YOLOv2 change line 19 "full_yolo" to False
To change to a different model architecture see lines 22-30 and change the ".cfg" file 
To change the weights see lines 22-30 change the ".weights" file or the number to a checkpoint in the ckpt folder
To change the threshold for YOLO change the threshold value in line 22-30 

------------
Other files that were not included in the demo

farneback.py 
Runs dense optical flow on the video on line 32

hogdetection.py
An attempt at using the built in HOG detector (very bad accuracy on our datasets)

train.py/yolo.py 
Part of our original attempt during which we thought about implementing yolo from scratch
