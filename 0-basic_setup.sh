
# Convert video to mp4 without audio
ffmpeg -i input/IMG_4656.MOV -vcodec h264 -vf scale=720:-2,setsar=1:1 -an input/input.mp4

# Get yolo dependencies
mkdir yolov3
wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names -P yolov3
wget https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg -P yolov3
wget https://pjreddie.com/media/files/yolov3.weights -P yolov3