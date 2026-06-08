from keras.models import load_model
import numpy as np

class GestureClassifier:
    def __init__(self):
        self.model = load_model("../02-dataset/gesture_recognition.keras")
        self.class_names = ["like", "dislike", "stop", "rock", "peace"]

    # Predict the gesture class from the cropped hand image.
    def predict(self, image):
        image = image.astype("float32") / 255.0
        image = np.expand_dims(image, axis=0)
        prediction = self.model.predict(image, verbose=0)
        class_index = np.argmax(prediction)
        return self.class_names[class_index]