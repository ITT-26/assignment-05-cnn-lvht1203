import cv2
import numpy as np

# AI-assisted: sepia filter implementation based on
# a standard OpenCV color transformation matrix.
def apply_sepia(frame):
    kernel = np.array([[0.272, 0.534, 0.131],[0.349, 0.686, 0.168],[0.393, 0.769, 0.189]])
    sepia = cv2.transform(frame, kernel)
    return np.clip(sepia, 0, 255).astype(np.uint8)

def zoom(frame, factor=1.5):
    height, width = frame.shape[:2]
    new_width = int(width / factor)
    new_height = int(height / factor)
    x = (width - new_width) // 2
    y = (height - new_height) // 2
    cropped = frame[y:y + new_height, x:x + new_width]
    return cv2.resize(cropped,(width, height))

# AI-assisted: decorative border rendering using OpenCV.
def draw_border(frame):
    cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (255, 0, 0), 8)
    return frame