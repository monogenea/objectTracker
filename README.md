# Detection and tracking in Python

<p align="center">
  <img src="https://user-images.githubusercontent.com/7695861/113501088-3590ad00-9523-11eb-81b3-448fd735375b.jpg?raw=true" alt="Example frame"/>
</p>

Part of the upcoming tutorial from my blog [poissonisfish](https://poissonisfish.com).

## Instructions

### Set up virtualenv
In a working directory of your choice, run `pip env python3 -m venv <ENV_NAME>`, then `source <ENV_NAME>/bin/activate`. Finally, install packages using `pip install -r requirements.txt`.

### Video conversion using the init Bash script

Create a MOV video recording and move it to the working directory. Run `./init.sh <PATH_TO_MOV>` to create subdirectories, convert the MOV video to MP4 and download the YOLOv3 dependencies for OpenCV.

### Run script

From the active environment, run `python tracker.py` and observe the frame-by-frame computations while the annotated output video is written.

## Acknowledgements

This work is inspired by a project assignment from the course Computer Vision I at [OpenCV.org](https://opencv.org).