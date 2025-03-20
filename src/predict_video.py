import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
from collections import deque
from model import Conv3DNet
import time
import argparse

class VideoPredictor:
    def __init__(self, config_path, model_path):
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self.model = Conv3DNet(
            num_classes=self.config['model']['num_classes'],
            input_channels=self.config['model']['input_channels']
        ).to(self.device)
        
        # Load model weights
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Initialize frame buffer
        self.frame_buffer = deque(maxlen=self.config['dataset']['clip_length'])
        self.class_names = self.config['dataset']['class_names']
        self.frame_size = tuple(self.config['dataset']['frame_size'])
    
    def preprocess_frame(self, frame):
        # Resize frame
        frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    def predict(self):
        if len(self.frame_buffer) < self.config['dataset']['clip_length']:
            return None, None
        
        # Convert frame buffer to tensor
        frames = np.array(list(self.frame_buffer))
        frames = torch.from_numpy(frames).float()
        # Normalize
        frames = frames / 255.0
        # Reshape to (1, C, T, H, W)
        frames = frames.permute(3, 0, 1, 2).unsqueeze(0)
        
        with torch.no_grad():
            frames = frames.to(self.device)
            outputs = self.model(frames)
            probs = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probs, dim=1)
            
            return self.class_names[prediction.item()], confidence.item()

def main():
    parser = argparse.ArgumentParser(description='Predict action from video')
    parser.add_argument('video_path', type=str, help='Path to the video file')
    args = parser.parse_args()
    
    # Initialize video predictor
    predictor = VideoPredictor(
        config_path='configs/config.yaml',
        model_path='checkpoints/best_model.pth'
    )
    
    # Open video file
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    output_path = f"{Path(args.video_path).stem}_predicted.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_thickness = 2
    
    print(f"Processing video: {args.video_path}")
    print(f"Output will be saved to: {output_path}")
    print("Press 'q' to quit")
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Preprocess frame
        processed_frame = predictor.preprocess_frame(frame)
        predictor.frame_buffer.append(processed_frame)
        
        # Get prediction
        action, confidence = predictor.predict()
        
        # Draw prediction on frame
        if action is not None:
            text = f"{action}: {confidence:.2f}"
            cv2.putText(frame, text, (10, 30), font, font_scale, (0, 255, 0), font_thickness)
        
        # Write frame to output video
        out.write(frame)
        
        # Display frame
        cv2.imshow('Action Recognition', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        frame_count += 1
        if frame_count % 30 == 0:  # Show progress every 30 frames
            print(f"Processed {frame_count} frames")
    
    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete! Output saved to {output_path}")

if __name__ == '__main__':
    main()
