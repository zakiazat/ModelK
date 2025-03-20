import os
import yaml
import argparse

def create_class_directories(config_path):
    """Create directory structure for the dataset."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create main directories
    train_path = config['dataset']['train_path']
    val_path = config['dataset']['val_path']
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(val_path, exist_ok=True)

    # Create class directories
    for action in config['dataset']['class_names']:
        # Training directories
        train_class_path = os.path.join(train_path, action)
        os.makedirs(train_class_path, exist_ok=True)
        
        # Validation directories
        val_class_path = os.path.join(val_path, action)
        os.makedirs(val_class_path, exist_ok=True)

    print("Created directory structure for the dataset.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create dataset directory structure')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    create_class_directories(args.config)
