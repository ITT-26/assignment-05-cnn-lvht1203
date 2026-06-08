# Exercise 02 – Dataset Collection and Gesture Recognition

## Requirements

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the notebook

Start Jupyter Lab:

```bash
jupyter lab
```

Open:

```text
02-dataset/gesture_evaluation.ipynb
```

Run all cells from top to bottom.

## Notes

* The trained model is stored in `gesture_recognition.keras`.
* The annotation file is `annot-thu.json`.
* The generated confusion matrix is saved as `conf-matrix.png`.


# Exercise 03 – Gesture Controlled Camera App

## Overview

This application combines MediaPipe hand tracking with the CNN gesture classifier trained in Assignment 02. The webcam image is analyzed in real time, and different hand gestures can be used to control camera functions without using the keyboard or mouse.

The application supports taking photos with a configurable countdown timer and several camera effects that can be controlled through hand gestures.

## Running the Application

Example:

```bash
python camera_app.py --time 5 --path photos
```

### Command Line Parameters

| Parameter | Description                              |
| --------- | ---------------------------------------- |
| `--time`  | Countdown duration before taking a photo |
| `--path`  | Folder where captured photos are saved   |

Examples:

```bash
python camera_app.py --time 3 --path captured_images
```

```bash
python camera_app.py --time 10 --path holiday_photos
```

---

## Supported Gestures

| Gesture    | Function                         |
| ---------- | -------------------------------- |
| Like       | Start countdown and take a photo |
| Stop       | Toggle sepia filter              |
| Peace      | Toggle zoom effect               |
| Rock       | Toggle decorative border         |
| Dislike    | Reset all active effects         |

---

## Additional Features

### Sepia Filter

Applies a vintage-style sepia effect to the webcam image.

### Zoom Effect

Zooms into the center region of the webcam image and scales it back to the original size.

### Decorative Border

Adds a visible border around the webcam image.

---

## Gesture Stabilization

Gesture predictions can occasionally fluctuate between neighboring classes. To reduce accidental activations, the application only accepts a gesture after it has been detected consistently across multiple consecutive frames.

This improves stability and reduces flickering of camera effects.

---

## Notes

* Photos are captured after the countdown reaches zero.
* Captured photos are saved using timestamp-based filenames.
* Camera effects can be combined (e.g. sepia + zoom + border).
* The captured photo itself does not include the active camera effects. The original webcam frame is saved.
* To toggle an effect off again, it may be necessary to briefly perform a different gesture before repeating the same gesture. This is a consequence of the gesture stabilization mechanism used to prevent repeated triggering.

---

## Development Files

The repository also contains `test_mediapipe.py`, which was used during development to verify MediaPipe hand detection, landmark visualization, and hand-region cropping before integrating these components into the final application.
