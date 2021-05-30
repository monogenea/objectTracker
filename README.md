# Object detection and tracking in Python

![](https://user-images.githubusercontent.com/7695861/113501088-3590ad00-9523-11eb-81b3-448fd735375b.jpg?raw=true)

Part of the upcoming tutorial from my blog [poissonisfish](https://poissonisfish.com).

# Instructions

- [x] Install Python version >= 3.7.0. If you don't have it, you can find it [here](https://www.python.org/downloads/).
      &nbsp;

### Set up `virtualenv`

- Open your terminal
  &nbsp;

- Install the `pipenv` tool by running the command

  - `pip install pipenv`
    &nbsp;

- You will now run the `venv` library module as a script which will spit out the result in the folder you'll define in `<VENV_PATH>`.
  In a directory of your choice (can be the current repo directory), run:

  - `python -m venv <VENV_PATH>`
    &nbsp;

- Once a virtual environment has been created, it can be “activated” using a script in the virtual environment’s binary directory. The invocation of the script is platform-specific (`<VENV_PATH>` is the path of the directory containing the virtual environment, which you previously created). Run:

| Platform | Shell           | Command to activate virtual environment   |
| -------- | --------------- | ----------------------------------------- |
| POSIX    | bash/zsh        | `$` source <VENV_PATH>/bin/activate       |
|          | fish            | `$` source <VENV_PATH>/bin/activate.fish  |
|          | csh/tcsh        | `$` source <VENV_PATH>/bin/activate.csh   |
|          | PowerShell Core | `$` /bin/Activate.ps1                     |
| Windows  | cmd.exe         | `C:\>` <VENV_PATH>\Scripts\activate.bat   |
|          | PowerShell      | `PS C:\>`<VENV_PATH>\Scripts\Activate.ps1 |

- Install required packages using

  - `pip install -r requirements.txt`
    &nbsp;

### Video conversion and object detection

- Create a `.mov` video recording and move it to the working directory and give it a name `<VIDEO_NAME>`.
  &nbsp;
- In order to create `Input` and `Output` directories, convert the `.mov` video to `.mp4` and download the `YOLOv3` dependencies for OpenCV, you should run:
  - `init.sh <VIDEO_NAME>`
    - `init.sh` uses `ffmpeg` command which you may not have yet but can get it [here](https://www.ffmpeg.org/download.html) for Linux, Windows or MacOS.
    - `init.sh` also contains commands such as `wget` which is a native Linux command to fetch web resources, if you're on Windows you should first [download wget.exe](https://eternallybored.org/misc/wget/) and add the executable directory path to your system environment PATH variable.
      If you're on MacOS, you should use a tool like HomeBrew.
      &nbsp;
- From the active environment, run `python tracker.py` and observe the frame-by-frame computations while the annotated output video is written.

### Acknowledgements

This work is inspired by a project assignment from the course Computer Vision I at [OpenCV.org](https://opencv.org).
