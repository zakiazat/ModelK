import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
from pathlib import Path
import random
import yaml

class ActionRecognitionDataset(Dataset):
    def __init__(self, root_dir, clip_length=16, frame_size=(224, 224), train=True, transform=None):
        """
        Args:
            root_dir (str): Directory with all the video folders
            clip_length (int): Number of frames per clip
            frame_size (tuple): Size to resize frames to (height, width)
            train (bool): Whether this is training or validation set
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.root_dir = Path(root_dir)
        self.clip_length = clip_length
        self.frame_size = frame_size
        self.train = train
        self.transform = transform
        
        # Load class names from config
        with open('configs/config.yaml', 'r') as f:
            config = yaml.safe_load(f)
            self.class_names = config['dataset']['class_names']
        
        # Get all video paths and their corresponding labels
        self.samples = []
        for class_path in self.root_dir.iterdir():
            if class_path.is_dir():
                class_name = class_path.name
                if class_name in self.class_names:
                    label = self.class_names.index(class_name)
                    for video_path in class_path.glob('*.mp4'):  # Adjust extension as needed
                        self.samples.append((str(video_path), label))
    
    def __len__(self):
        return len(self.samples)
    
    def load_video(self, video_path):
        """Load video and return frames."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if self.train:
            # Random starting point for training
            start_frame = random.randint(0, max(0, total_frames - self.clip_length))
        else:
            # Center clip for validation
            start_frame = max(0, (total_frames - self.clip_length) // 2)
        
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read frames
        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (self.frame_size[1], self.frame_size[0]))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            else:
                # If video is too short, duplicate last frame
                if frames:
                    frames.append(frames[-1])
                else:
                    # If no frames were read, create empty frame
                    empty_frame = np.zeros((self.frame_size[0], self.frame_size[1], 3), dtype=np.uint8)
                    frames.append(empty_frame)
        
        cap.release()
        
        # Ensure we have exactly clip_length frames
        frames = frames[:self.clip_length]
        while len(frames) < self.clip_length:
            frames.append(frames[-1])
        
        return np.array(frames)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # Load video frames
        frames = self.load_video(video_path)
        
        # Convert to torch tensor and reshape to (C, T, H, W)
        frames = torch.from_numpy(frames).float()
        frames = frames.permute(3, 0, 1, 2)  # (T, H, W, C) -> (C, T, H, W)
        frames = frames / 255.0  # Normalize to [0, 1]
        
        if self.transform:
            frames = self.transform(frames)
        
        return frames, label
