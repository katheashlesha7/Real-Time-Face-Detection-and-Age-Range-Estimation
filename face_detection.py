"""Detect faces from a webcam and estimate an age range for each face."""

import argparse
import sys
from pathlib import Path

import cv2


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
AGE_MODEL = MODELS_DIR / "age_net.caffemodel"
AGE_CONFIG = MODELS_DIR / "age_deploy.prototxt"

CAMERA_INDEX = 0
WINDOW_NAME = "Face Detection and Age Estimate"
CASCADE_FILENAME = "haarcascade_frontalface_default.xml"

# The age model predicts one of these eight broad ranges.
AGE_RANGES = ("0-2", "4-6", "8-12", "15-20", "25-32", "38-43", "48-53", "60-100")
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
GREEN = (0, 255, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect faces in webcam video and estimate an age range."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CAMERA_INDEX,
        help=f"camera device index (default: {CAMERA_INDEX})",
    )
    return parser.parse_args()


def predict_age_range(age_net, face_image) -> str:
    """Return the broad age bucket with the highest model score."""
    blob = cv2.dnn.blobFromImage(
        face_image,
        scalefactor=1.0,
        size=(227, 227),
        mean=MODEL_MEAN_VALUES,
        swapRB=False,
        crop=False,
    )
    age_net.setInput(blob)
    scores = age_net.forward().flatten()
    return AGE_RANGES[int(scores.argmax())]


def main() -> int:
    args = parse_args()

    if not AGE_CONFIG.is_file() or not AGE_MODEL.is_file():
        print(
            "Age model files are missing. From this project folder, run "
            "'python download_age_model.py' first.",
            file=sys.stderr,
        )
        return 1

    cascade_path = cv2.data.haarcascades + CASCADE_FILENAME
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print(f"Could not load the face detector: {cascade_path}", file=sys.stderr)
        return 1

    try:
        age_net = cv2.dnn.readNetFromCaffe(str(AGE_CONFIG), str(AGE_MODEL))
    except cv2.error as error:
        print(f"Could not load the age model: {error}", file=sys.stderr)
        return 1

    video_capture = cv2.VideoCapture(args.camera)
    if not video_capture.isOpened():
        print(
            f"Could not open camera {args.camera}. Check camera access or try "
            "another index with --camera.",
            file=sys.stderr,
        )
        return 1

    try:
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Could not read a frame from the camera.", file=sys.stderr)
                return 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )

            for (x, y, width, height) in faces:
                face_image = frame[y : y + height, x : x + width]
                age_range = predict_age_range(age_net, face_image)

                cv2.rectangle(
                    frame,
                    (x, y),
                    (x + width, y + height),
                    GREEN,
                    2,
                )

                label = f"Age: {age_range}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                label_top = max(0, y - label_height - baseline - 8)
                label_left = min(x, frame.shape[1] - label_width - 8)
                cv2.rectangle(
                    frame,
                    (label_left, label_top),
                    (
                        label_left + label_width + 8,
                        label_top + label_height + baseline + 8,
                    ),
                    GREEN,
                    cv2.FILLED,
                )
                cv2.putText(
                    frame,
                    label,
                    (label_left + 4, label_top + label_height + 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        # Ctrl+C in the terminal also stops the camera loop cleanly.
        pass
    finally:
        video_capture.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

