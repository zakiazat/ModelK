import os
import yaml
import argparse
from tqdm import tqdm
import yt_dlp

def download_video(url, output_path):
    """Download video from URL."""
    ydl_opts = {
        'format': 'best[ext=mp4][height>=224]',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': False
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"Successfully downloaded {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        print(f"Failed to download {url}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download videos from URLs')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--urls_file', type=str, required=True, help='Path to URLs file')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Load URLs
    with open(args.urls_file, 'r') as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    videos = []
    for line in lines:
        url, category, split, name = line.split(',')
        videos.append({
            'url': url,
            'category': category,
            'split': split,
            'name': name
        })

    # Validate categories
    valid_categories = config['dataset']['class_names']
    for video in videos:
        if video['category'] not in valid_categories:
            raise ValueError(f"Invalid category {video['category']}. Must be one of {valid_categories}")

    # Download videos
    print("Downloading videos:")
    for video in tqdm(videos):
        # Determine output path
        base_dir = config['dataset']['train_path'] if video['split'] == 'train' else config['dataset']['val_path']
        output_dir = os.path.join(base_dir, video['category'])
        output_path = os.path.join(output_dir, f"{video['name']}.mp4")

        # Skip if video already exists
        if os.path.exists(output_path):
            print(f"Video {output_path} already exists, skipping...")
            continue

        # Download video
        download_video(video['url'], output_path)

if __name__ == '__main__':
    main()
