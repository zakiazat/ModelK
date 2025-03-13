import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_videos
from model import build_conv3d_model
import os
import h5py
import tensorflow as tf
import cv2

# Load dataset
X, y = load_videos("dataset")
print("Dataset shape:", X.shape, "Labels shape:", y.shape)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build model
input_shape = (20, 64, 64, 3)  # (num_frames, height, width, channels)
num_classes = len(os.listdir("dataset"))
model = build_conv3d_model(input_shape, num_classes)

# Train model
model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_test, y_test))

# Save the trained model
model.save("models/action_model.h5")

# Evaluate model
# Open the HDF5 file
file_path = "models/action_model.h5"
with h5py.File(file_path, "r") as f:
    # Print all root level groups
    print("Root level groups:", list(f.keys()))

    # Iterate through all groups and datasets
    def print_attrs(name, obj):
        print(f"{name}: {obj}")

    f.visititems(print_attrs)

# Load the model
model = tf.keras.models.load_model("models/action_model.h5")

# Print the model summary
model.summary()

# Export video predictions
def display_predictions(model, video_path, input_shape):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_buffer = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_resized = cv2.resize(frame, (input_shape[2], input_shape[1]))
        frame_normalized = frame_resized / 255.0
        frame_buffer.append(frame_normalized)

        # Ensure the buffer has the correct number of frames
        if len(frame_buffer) == input_shape[0]:
            frame_sequence = np.expand_dims(frame_buffer, axis=0)

            # Make prediction
            prediction = model.predict(frame_sequence)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Display the frame with the predicted label
            label = f"Predicted: {predicted_class}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Video", frame)

            # Remove the first frame from the buffer
            frame_buffer.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
video_path = "dataset/jumping/jumping.mp4"
display_predictions(model, video_path, input_shape)