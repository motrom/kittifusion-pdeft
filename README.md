# Overview/Status
This code is for the paper "Vehicular Multi-Object Tracking with Persistent Detector Failures", currently in review for IEEE Transactions on Intelligent Vehicles, as well as the thesis "Practical Probabilistic Multi-Object Tracking for Autonomous Vehicles". A preprint of the former is at https://arxiv.org/abs/1907.11306. The latter will be online by January.  
Specifically, this code performs vehicle tracking using a single camera and vertically sparse lidar. https://github.com/motrom/kittitracking-dpeft has similar code for dense lidar, which has been organized for decently use use or modification. This repository is primarily online so that all experiments from the publications are reproducible, and is not very clean.

# Description
Vehicle tracking on the Kitti dataset. A deep monocular vision detector https://github.com/Zengyi-Qin/MonoGRNet is fused with a handmade lidar segmentation detector. MonoGRNet was trained w/o any images from the tracking scenes that were tested; see the Input section to acquire the resulting detections.  
Each vehicle is tracked as a 2D rectangle (bird's eye view) with a simple bicycle motion model. The tracker is mostly the same as in the kittitracking-pdeft repository, with three major modifications for handling these particular detectors. The model for the MonoGRNet detector is augmented with a bias term in the estimation of distance from the vehicle (it was observed that this detector under/overestimates distance of vehicles from the camera by around 10%). Additionally, there are many non-vehicle objects detected by lidar: rather than track all potentially "fake" objects, only objects with a MonoGRNet detection are tracked. Finally, hypothesis-oriented multiple-hypothesis tracking is used rather than single-hypothesis tracking.

# Usage
## Dependencies
1. Python (python 3.6 has been used primarily, but 2.7 should be fine)
2. numpy & scipy, imageio & matplotlib for visualizations
3. OpenCV version 4
4. numba (only used to speed up some functions, can be disabled easily if you don't want to install numba)
5. github.com/motrom/fastmurty for MHT data association (follow instructions in that repo to build, then link to the resulting file in mhtdaClink.py)

## Input
MonoGRNet detections are available at https://utexas.box.com/s/gm4lw81k8cu53c0ty87xx5rw7wmrqwai.  
The tracker also uses Kitti's lidar and positioning data. For visualization and performance evaluation, it needs the left images and the kitti annotation text files. All data files are specified by a file in runconfigs/ (currently there is an example.py) using `somestring.format(scene_number)` or `somestring.format(scene_number, frame_number)`, so make appropriate modifications to each `somestring` for your file organization.  
Finally, the code currently only handles the first ten scenes from the kitti training set. The detections are available for these scenes, other scenes were partially used while training the monocular detector.

## Running
First, set up a file in runconfigs/ that specifies the data to test on. Ground planes for a scene and timestep must be created via ground.py (as in the kittitracking-peft repo). track2.py can then be run on any scene, and saves and/or visualizes results. Saved results can be evaluated with evaluate.py.  
The design and performance of the lidar segmentation detector can be examined visually using the three lidarviz files.

# Performance
The submitted paper shows a clear performance difference between tracking with only monocular detections and fusion with lidar segments. However, the result is still well below the performance of trackers using dense lidar. Image-detected vehicles are frequently "matched" with lidar detections of incorrect objects, especially for distant lines of parked cars. The thesis chapter 5 explores multiple-hypothesis tracking as a means of improving matches, but we didn't obtain a significant improvement.

# Acknowledgements
Using code from https://github.com/utiasSTARS/pykitti/blob/master/pykitti/odometry.py to obtain the host vehicle's motion.  
Included kitti's 2D benchmark code along with the requisite assignment algorithm code.
