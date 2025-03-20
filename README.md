# Conv3D Action Recognition

A deep learning project for recognizing human actions in videos using 3D Convolutional Neural Networks (Conv3D).

## Features

- Real-time action recognition from videos
- Support for both video file and webcam input
- Pre-trained model for common actions
- Easy-to-use training pipeline
- Real-time visualization of predictions

## Project Structure

```
conv3d_training/
├── configs/              # Configuration files
│   └── config.yaml       # Main configuration file
├── src/                  # Source code
│   ├── model.py         # Conv3D model architecture
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── train.py         # Training script
│   ├── predict_video.py # Video prediction script
│   └── predict_webcam.py# Webcam prediction script
└── requirements.txt      # Python dependencies
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
- Place training videos in `data/train/<action_name>/`
- Place validation videos in `data/val/<action_name>/`

3. Configure the model:
- Edit `configs/config.yaml` to match your dataset

## Training

To train the model:
```bash
python src/train.py --config configs/config.yaml
```

## Prediction

To run predictions on a video:
```bash
python src/predict_video.py <path_to_video>
```

To run predictions using webcam:
```bash
python src/predict_webcam.py
```

## Model Architecture

The model uses a 3D CNN architecture optimized for video understanding:
- Input: Video clips (sequence of frames)
- Conv3D layers for spatiotemporal feature extraction
- Global average pooling
- Fully connected layers for classification

## License

MIT License
