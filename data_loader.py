import numpy as np
import cv2
import os

def load_videos(directory, frame_size=(64, 64), num_frames=20):
    video_data = []
    labels = []
    class_labels = os.listdir(directory)
    
    for label, class_name in enumerate(class_labels):
        class_path = os.path.join(directory, class_name)
        
        for video_name in os.listdir(class_path):
            video_path = os.path.join(class_path, video_name)
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while len(frames) < num_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.resize(frame, frame_size)
                frame = frame / 255.0  # Normalize pixel values
                frames.append(frame)
            
            cap.release()
            
            if len(frames) == num_frames:
                video_data.append(np.array(frames))
                labels.append(label)
    
    return np.array(video_data), np.array(labels)