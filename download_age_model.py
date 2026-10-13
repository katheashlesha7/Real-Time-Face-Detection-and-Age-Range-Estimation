"""Download the age-estimation model files used by face_detection.py."""

from pathlib import Path
from urllib.request import urlopen
import os
import sys


PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
FILES = {
    "age_deploy.prototxt": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/"
        "master/age_net_definitions/deploy.prototxt"
    ),
    "age_net.caffemodel": (
        "https://raw.githubusercontent.com/GilLevi/AgeGenderDeepLearning/"
        "master/models/age_net.caffemodel"
    ),
}


def download_file(filename: str, url: str) -> None:
    destination = MODELS_DIR / filename
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading {filename}...")
    try:
        with urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                output.write(chunk)
        if temporary.stat().st_size == 0:
            raise OSError(f"The download of {filename} was empty.")
        os.replace(temporary, destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for filename, url in FILES.items():
            download_file(filename, url)
    except Exception as error:
        print(f"Could not download model files: {error}", file=sys.stderr)
        return 1

    print(f"Model files saved in: {MODELS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
