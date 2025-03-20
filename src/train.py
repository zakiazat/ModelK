import os
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import Conv3DNet
from dataset import ActionRecognitionDataset

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    logging.info(f"\n{'='*50}")
    logging.info(f"Training Epoch {epoch}")
    logging.info(f"{'='*50}")
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (inputs, targets) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
        # Update progress bar with more details
        pbar.set_postfix({
            'Loss': f"{running_loss/(batch_idx+1):.4f}",
            'Acc': f"{100.*correct/total:.2f}%",
            'Correct': f"{correct}/{total}"
        })
    
    epoch_loss = running_loss/len(train_loader)
    epoch_acc = 100.*correct/total
    
    # Log metrics
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_acc, epoch)
    
    logging.info(f"Training - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device, epoch, writer):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    logging.info(f"\n{'-'*50}")
    logging.info(f"Validation Epoch {epoch}")
    logging.info(f"{'-'*50}")
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation')
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{running_loss/(batch_idx+1):.4f}",
                'Acc': f"{100.*correct/total:.2f}%",
                'Correct': f"{correct}/{total}"
            })
    
    val_loss = running_loss/len(val_loader)
    val_acc = 100.*correct/total
    
    # Log metrics
    writer.add_scalar('Loss/val', val_loss, epoch)
    writer.add_scalar('Accuracy/val', val_acc, epoch)
    
    logging.info(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    return val_loss, val_acc

def main():
    parser = argparse.ArgumentParser(description='Train Conv3D model for action recognition')
    parser.add_argument('--config', type=str, required=True, help='path to config file')
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create directories
    save_dir = Path(config['training']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Create model
    model = Conv3DNet(
        num_classes=config['model']['num_classes'],
        input_channels=config['model']['input_channels']
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=5, verbose=True
    )
    
    # TensorBoard writer
    writer = SummaryWriter(log_dir=str(save_dir / 'logs'))
    
    # Create data loaders
    train_dataset = ActionRecognitionDataset(
        root_dir=config['dataset']['train_path'],
        clip_length=config['dataset']['clip_length'],
        frame_size=tuple(config['dataset']['frame_size'])
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=4
    )

    val_dataset = ActionRecognitionDataset(
        root_dir=config['dataset']['val_path'],
        clip_length=config['dataset']['clip_length'],
        frame_size=tuple(config['dataset']['frame_size'])
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=4
    )

    # Training loop
    best_val_acc = 0
    for epoch in range(config['training']['epochs']):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, writer
        )
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, device, epoch, writer
        )
        logging.info(f'Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, save_dir / 'best_model.pth')
            logging.info(f"Saved new best model with validation accuracy: {val_acc:.2f}%")
        
        # Update learning rate
        scheduler.step(val_loss)
    
    writer.close()

if __name__ == '__main__':
    main()
