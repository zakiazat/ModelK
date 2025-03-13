import numpy as np
import cv2
import tensorflow as tf
import os

def predict_action(video_path, model_path):
    model = tf.keras.models.load_model(model_path)

    cap = cv2.VideoCapture(video_path)
    frames = []

    while len(frames) < 20:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (64, 64)) / 255.0
        frames.append(frame)

    cap.release()

    if len(frames) == 20:
        frames = np.array(frames).reshape(1, 20, 64, 64, 3)
        prediction = model.predict(frames)
        class_labels = os.listdir("dataset")
        return class_labels[np.argmax(prediction)]

    return "Invalid video"

# Test with a new video
print("Predicted Action:", predict_action("test_video.mp4", "models/action_model.h5"))