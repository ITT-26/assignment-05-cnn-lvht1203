import cv2
import mediapipe as mp

"""
Small test script used during development.

This script verifies that MediaPipe can detect a hand from
the webcam and correctly draw hand landmarks before
integrating the functionality into camera_app.py.
"""
# Initialize MediaPipe hand tracking and drawing utilities.
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Open the default webcam.
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # MediaPipe expects RGB images instead of OpenCV's BGR format.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        h, w, _ = frame.shape
        for hand in results.multi_hand_landmarks:
            # Draw the detected hand landmarks and connections.
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
            xs = []
            ys = []
            for lm in hand.landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))
            # Create a bounding box around all detected landmarks.
            x1 = max(min(xs) - 20, 0)
            y1 = max(min(ys) - 20, 0)
            x2 = min(max(xs) + 20, w)
            y2 = min(max(ys) + 20, h)
            cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
            # Crop the detected hand region for later use
            # in the gesture classification pipeline.
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop = cv2.resize(crop, (64, 64))
                cv2.imshow("Crop", crop)
                
    cv2.imshow("Hand Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27 or key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()