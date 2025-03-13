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
model.save("models/action_model.keras")

# Load the trained model
model = tf.keras.models.load_model("models/action_model.keras")

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")

# Make predictions on the test set
predictions = model.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)

# Display predictions along with actual labels
for i in range(len(X_test)):
    print(f"Actual: {y_test[i]}, Predicted: {predicted_classes[i]}")

# Function to display predictions in video form
def display_predictions(model, input_shape, class_labels):
    cap = cv2.VideoCapture(0)  # Use 0 for the default webcam
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Create a named window with HD resolution
    cv2.namedWindow("Video", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Video", 1280, 720)

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
            predicted_label = class_labels[predicted_class]

            # Display the frame with the predicted label
            label = f"Predicted: {predicted_label}"
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Highlight detected movements
            cv2.rectangle(frame, (10, 10), (frame.shape[1] - 10, frame.shape[0] - 10), (0, 255, 0), 2)
            cv2.imshow("Video", frame)

            # Remove the first frame from the buffer
            frame_buffer.pop(0)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
class_labels = ["no movement", "moving", "jumping"]  # Replace with your actual class labels
display_predictions(model, input_shape, class_labels)