# Real-Time Face Detection and Age Range Estimation

A Python webcam application that detects faces, draws a green rectangle around each detected face, and displays a predicted age range above it. Face detection uses OpenCV's Haar cascade; age estimation uses an age classification model through OpenCV's DNN module.

The age estimate is an approximate visual prediction. It is not a person's verified or actual age. This app detects faces; it does not identify people or recognize their identities.

## Features

- Processes live video from a webcam.
- Draws a green box around each detected face.
- Shows one of eight predicted age ranges: `0-2`, `4-6`, `8-12`, `15-20`, `25-32`, `38-43`, `48-53`, or `60-100`.
- Press `q` in the video window to quit. Press Ctrl+C in the terminal as another way to stop it.
- Does not save camera frames or predictions to disk.

## Requirements

- Python 3
- A webcam
- Internet access for the one-time model download
- The `opencv-python` package

## Setup

Open a terminal in the project folder and create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- Windows Command Prompt: `.venv\Scripts\activate.bat`
- macOS or Linux: `source .venv/bin/activate`

Install OpenCV:

```bash
python -m pip install -r requirements.txt
```

Download the age model files once:

```bash
python download_age_model.py
```

The model weight file is large (about 44 MB). The download script places it under `models/`. The weight file is excluded from Git by `.gitignore`.

## Run

```bash
python face_detection.py
```

Use a different camera if needed:

```bash
python face_detection.py --camera 1
```

The default camera index is `0`. Press `q` while the video window is focused to quit.

## Project files

```text
face_detection.py       Webcam face detection and age-range display
download_age_model.py  Downloads the age model files into models/
requirements.txt         Python dependency list
.gitignore               Ignores Python caches, virtual environments, and model weights
README.md                Project documentation
models/                  Created by the download script; contains the age model files
```

## How it works

1. OpenCV captures a webcam frame and converts it to grayscale.
2. OpenCV's bundled `haarcascade_frontalface_default.xml` detects face regions.
3. Each face crop is passed to the age classification network.
4. The app selects the highest-scoring age bucket and draws it with a green face rectangle.

The detector and age estimator are separate components. Age estimation only runs for regions the face detector has found.

## Troubleshooting

- **Age model files are missing:** Run `python download_age_model.py` from the project folder and wait for it to finish.
- **Download fails:** Check the internet connection, then run `python download_age_model.py` again.
- **Camera cannot be opened:** Close apps that may be using the webcam, check camera permissions, or try another camera number with `--camera 1`.
- **`ModuleNotFoundError: No module named 'cv2'`:** Activate the virtual environment and run `python -m pip install -r requirements.txt`.
- **PowerShell blocks activation:** Run `.venv\Scripts\python.exe -m pip install -r requirements.txt`, then `.venv\Scripts\python.exe download_age_model.py`, and `.venv\Scripts\python.exe face_detection.py`.

## Accuracy and responsible use

The age network returns broad ranges, not exact ages. Lighting, camera quality, face angle, occlusion, and the training data can affect its predictions. It can be wrong and may perform unevenly across people. Do not use its output for age verification, access decisions, or other consequential decisions. Get consent before using a camera to analyze people.

## Model reference

The age classifier is from the work by Gil Levi and Tal Hassner, *Age and Gender Classification Using Convolutional Neural Networks*, CVPR Workshop on Analysis and Modeling of Faces and Gestures, 2015. See the [original model repository](https://github.com/GilLevi/AgeGenderDeepLearning) for the model and research details. Review the model source's terms before redistribution or public/commercial use.

## License

No license has been selected for this project yet. Add a project license before distributing or reusing it publicly.

