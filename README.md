# Object detection and tracking in Python

![](https://user-images.githubusercontent.com/7695861/113501088-3590ad00-9523-11eb-81b3-448fd735375b.jpg?raw=true)

Part of the upcoming tutorial from my blog [poissonisfish](https://poissonisfish.com) featuring an example of my own.

# Instructions

- [x] Install Python >= 3.7.0, available [here](https://www.python.org/downloads/)
- [x] Clone this repository to a local / remote machine

### Set up `virtualenv`

From a new terminal, after setting the repository as the working directory:

- Install `pipenv` by running `pip install pipenv`

- Run `pip env python3 -m venv <ENV_NAME>` / `python -m venv <ENV_NAME>` to create the `pip` environment

- Activate the `pip` environment using one of the following methods:

| Platform | Shell           | Command to activate virtual environment   |
| -------- | --------------- | ----------------------------------------- |
| POSIX    | bash/zsh        | `$` source <ENV_NAME>/bin/activate       |
|          | fish            | `$` source <ENV_NAME>/bin/activate.fish  |
|          | csh/tcsh        | `$` source <ENV_NAME>/bin/activate.csh   |
|          | PowerShell Core | `$` /bin/Activate.ps1                     |
| Windows  | cmd.exe         | `C:\>` <ENV_NAME>\Scripts\activate.bat   |
|          | PowerShell      | `PS C:\>`<ENV_NAME>\Scripts\Activate.ps1 |

- Install required packages using `pip install -r requirements.txt`

### Video conversion and object detection

- Create a `.mov` video recording and move it to the working directory

- Run `./init.sh <PATH_TO_MOV>` to create `input` and `output` directories, convert the `.MOV` video to `.mp4` and download the `YOLOv3` dependencies for OpenCV. Please note this script requires the following utilities:
    - `ffmpeg`, available [here](https://www.ffmpeg.org/download.html) for Linux, Windows and macOS
    - `wget`, readily available for Linux; Windows users can [download wget.exe here](https://eternallybored.org/misc/wget/) and add the executable path to the system environment PATH variable; macOS users can get it from HomeBrew or similar

- From the active environment, run `python tracker.py` and watch frame-by-frame computations while the annotated output video is written

### Acknowledgements

This work is inspired by a project assignment from the course Computer Vision I at [OpenCV.org](https://opencv.org).
