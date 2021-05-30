#!/bin/bash 

# Create subdirs
mkdir yolov3 input output

# Convert video (parse argument) to 720p mp4 without audio
echo "Converting $1 to MP4..."
ffmpeg -i $1 -vcodec h264 -vf scale=720:-2,setsar=1:1 -an input/input.mp4

# Get yolo dependencies
echo "Installing YOLOv3 dependencies..."
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P yolov3
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P yolov3
wget https://pjreddie.com/media/files/yolov3.weights -P yolov3