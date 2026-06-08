import cv2
import mediapipe as mp
import time
import argparse
import os

from datetime import datetime
from gesture_classifier import GestureClassifier
from camera_features import (apply_sepia, zoom, draw_border)

class CameraApp:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--time", type=int, default=3)
        parser.add_argument("--path", type=str, default="captured_images")
        self.args = parser.parse_args()

        # Load the CNN gesture classifier trained in the previous task.
        self.classifier = GestureClassifier()

        # Open webcam and initialize MediaPipe hand tracking.
        self.cap = cv2.VideoCapture(0)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.drawer = mp.solutions.drawing_utils

        # Current state of the camera effects.
        self.sepia_enabled = False
        self.zoom_enabled = False
        self.border_enabled = False

        # Values used to avoid triggering the same gesture too often.
        self.last_gesture = ""
        self.last_trigger_time = 0
        self.cooldown = 1
        self.last_detected_gesture = None
        self.pending_gesture = None
        self.pending_count = 0
        self.required_frames = 4

    def countdown(self):
        # AI-assisted: show the selfie timer directly in the OpenCV window.
        for i in range(self.args.time, 0, -1):
            success, frame = self.cap.read()
            if not success:
                return      
            cv2.putText(frame,str(i),(280, 250),cv2.FONT_HERSHEY_SIMPLEX,5,(0, 0, 255),8)
            cv2.imshow("Gesture Camera",frame)
            cv2.waitKey(1)
            time.sleep(1)
    
    def save_photo(self, frame):
        # AI-assisted: save the captured image with a timestamp-based filename.
        os.makedirs(self.args.path, exist_ok=True)
        filename = datetime.now().strftime("%Y%m%d_%H%M%S.jpg")
        cv2.imwrite(os.path.join(self.args.path, filename),frame)
        print("Saved:", filename)

    def handle_gesture(self, gesture, frame):
        # Map stable gesture predictions to camera actions.
        current_time = time.time()

        # Avoid repeated triggering while the same gesture is still being held.
        if (gesture == self.last_gesture and current_time - self.last_trigger_time < self.cooldown):
            return
        
        # Gesture mapping:
        # like    -> countdown and take photo
        # stop    -> toggle sepia filter
        # peace   -> toggle zoom
        # rock    -> toggle decorative border
        # dislike -> reset all effects
        if gesture == "like":
            self.countdown()
            success, frame = self.cap.read()
            if success:
                self.save_photo(frame)
                cv2.putText(frame,"PHOTO SAVED",(120, 80),cv2.FONT_HERSHEY_SIMPLEX,1.5,(0, 255, 0),3)
                cv2.imshow("Gesture Camera",frame)
                cv2.waitKey(500)
            self.last_trigger_time = current_time
        elif gesture == "stop":
            self.sepia_enabled = (not self.sepia_enabled)
        elif gesture == "peace":
            self.zoom_enabled = (not self.zoom_enabled)
        elif gesture == "rock":
            self.border_enabled = (not self.border_enabled) 
        elif gesture == "dislike":
            self.sepia_enabled = False
            self.zoom_enabled = False
            self.border_enabled = False
        self.last_gesture = gesture
        self.last_trigger_time = current_time

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            # MediaPipe expects RGB images, while OpenCV uses BGR.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb)

            if results.multi_hand_landmarks:
                h, w, _ = frame.shape
                for hand in results.multi_hand_landmarks:
                    self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    # Build a bounding box from all detected hand landmarks.
                    xs = []
                    ys = []
                    for lm in hand.landmark:
                        xs.append(int(lm.x * w))
                        ys.append(int(lm.y * h))
                    x1 = max(min(xs) - 20,0)
                    y1 = max(min(ys) - 20,0)
                    x2 = min(max(xs) + 20, w)
                    y2 = min(max(ys) + 20, h)
                    cv2.rectangle(frame,(x1, y1),(x2, y2),(0, 255, 0),2)
                    # Crop the detected hand and resize it to the same input size that was used when training the CNN model.
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        crop = cv2.resize(crop,(64, 64))
                        gesture = (self.classifier.predict(crop))
                        # Only accept a gesture after several consecutive frames.
                        # This reduces accidental triggers caused by unstable predictions.
                        if gesture == self.pending_gesture:
                            self.pending_count += 1
                        else:
                            self.pending_gesture = gesture
                            self.pending_count = 1
                        if (self.pending_count >= self.required_frames and gesture != self.last_detected_gesture):
                            self.handle_gesture(gesture, frame)
                            self.last_detected_gesture = gesture
                            self.pending_count = 0
                        cv2.putText(frame,gesture,(30, 50),cv2.FONT_HERSHEY_SIMPLEX,1,(0, 255, 0),2)

            # Apply the currently enabled camera effects before displaying the frame.
            if self.sepia_enabled:
                frame = apply_sepia(frame)

            if self.zoom_enabled:
                frame = zoom(frame)

            if self.border_enabled:
                frame = draw_border(frame)
            cv2.imshow("Gesture Camera",frame)

            key = cv2.waitKey(1)
            if key == ord("q") or key == 27:
                break

        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = CameraApp()
    app.run()