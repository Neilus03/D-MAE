import torch
import torch.nn as nn
import torch.optim as optim
import sys, os, yaml
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
import io
import safetensors.torch  # Added for safetensors support
from safetensors import safe_open
from PIL import Image


sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from data.dataloader import get_dataloaders, denormalize_RGB, denormalize_depth
from model.dmae import MAE, mae_loss, extract_patches, assemble_patches_with_gaps
from utils.visualization_utils import unpatchify, extract_patches, assemble_patches_with_gaps, log_visualizations

# Load configuration
config_path = '/home/ndelafuente/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize WandB
wandb.init(project=config['logging']['wandb_project'], entity=config['logging']['wandb_entity'])


def train_epoch(model, dataloader, optimizer, epoch, device):
    '''
    Runs one epoch of training.
    Args:
        model: DMAE model
        dataloader: training dataloader
        optimizer: optimizer
        epoch: current epoch
        device: device to run the model
    Returns:
        avg_loss: average loss over the epoch
        avg_rgb_loss: average RGB loss over the epoch
        avg_depth_loss: average depth loss over the epoch
    '''
    model.train()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    for batch_idx, (image_depths, patches) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}")):
        image_depths = image_depths.to(device)
        patches = patches.to(device)  # Move patches to device

        optimizer.zero_grad()

        # Forward pass
        pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched, x_masked = model(image_depths)

        # Flatten patches to match pred's shape
        target = patches.view(patches.size(0), patches.size(1), -1)  # Shape: [batch_size, num_patches, patch_size*patch_size*num_channels]

        # Compute loss
        loss, loss_rgb, loss_depth = mae_loss(pred, target, mask)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_rgb_loss += loss_rgb.item()
        total_depth_loss += loss_depth.item()

        # Log training metrics
        if batch_idx % config['logging']['loss_log_interval'] == 0:
            wandb.log({
                'Train Loss': loss.item(),
                'Train RGB Loss': loss_rgb.item(),
                'Train Depth Loss': loss_depth.item(),
                'Epoch': epoch,
                'Batch': batch_idx
            })
        
        # Visualizations
        if batch_idx % config['logging']['train_viz_log_interval'] == 0:
            log_visualizations(image_depths, reconstructed_image, reconstructed_depth, mask, epoch, batch_idx, prefix='Train')
            
    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    print(f"Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg RGB Loss: {avg_rgb_loss:.4f}, Avg Depth Loss: {avg_depth_loss:.4f}")
    return avg_loss, avg_rgb_loss, avg_depth_loss

def validate_epoch(model, dataloader, epoch, device):
    '''
    Runs one epoch of validation.
    Args:
        model: DMAE model
        dataloader: validation dataloader
        epoch: current epoch
        device: device to run the model
    Returns:
        avg_loss: average loss over the epoch
        avg_rgb_loss: average RGB loss over the epoch
        avg_depth_loss: average depth loss over the epoch
    '''
    model.eval()
    total_loss = 0
    total_rgb_loss = 0
    total_depth_loss = 0
    with torch.no_grad():
        for batch_idx, (image_depths, patches) in enumerate(tqdm(dataloader, desc=f"Validation Epoch {epoch}")):
            image_depths = image_depths.to(device)
            patches = patches.to(device)  # Move patches to device

            # Forward pass
            pred, mask, ids_restore, reconstructed_image, reconstructed_depth, x_patched, x_masked = model(image_depths)

            # Flatten patches to match pred's shape
            target = patches.view(patches.size(0), patches.size(1), -1)  # Shape: [batch_size, num_patches, patch_size*patch_size*num_channels]

            # Compute loss
            loss, loss_rgb, loss_depth = mae_loss(pred, target, mask)

            total_loss += loss.item()
            total_rgb_loss += loss_rgb.item()
            total_depth_loss += loss_depth.item()

            # Log validation visualizations for the first batch
            if batch_idx == 0:
                log_visualizations(image_depths, reconstructed_image, reconstructed_depth, mask, epoch, batch_idx, prefix='Validation')
    
    avg_loss = total_loss / len(dataloader)
    avg_rgb_loss = total_rgb_loss / len(dataloader)
    avg_depth_loss = total_depth_loss / len(dataloader)
    print(f"Validation Epoch {epoch} - Avg Loss: {avg_loss:.4f}, Avg RGB Loss: {avg_rgb_loss:.4f}, Avg Depth Loss: {avg_depth_loss:.4f}")
    return avg_loss, avg_rgb_loss, avg_depth_loss

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader = get_dataloaders(config)

    # Initialize model
    d_model = config['model']['d_model']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']
    n_channels = config['model']['n_channels']
    
    # Initialize model
    model = MAE(d_model, img_size, patch_size, n_channels)
    model.to(device)
    
    
    # Load model if checkpoint exists
    checkpoint_dir = config['logging']['model_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, 'best_model.safetensors')
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        state_dict = safetensors.torch.load_file(checkpoint_path)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = state_dict['epoch'] + 1
        best_val_loss = state_dict['best_val_loss']
        print(f"Loaded checkpoint from {checkpoint_path}, starting from epoch {start_epoch}")
    
    # Log model parameters
    wandb.log({'Number of Parameters': count_parameters(model)})
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])

    num_epochs = config['training']['num_epochs']

    for epoch in range(start_epoch, num_epochs):
        # Training
        train_loss, train_rgb_loss, train_depth_loss = train_epoch(model, train_loader, optimizer, epoch, device)

        # Validation
        val_loss, val_rgb_loss, val_depth_loss = validate_epoch(model, val_loader, epoch, device)

        # Log epoch metrics
        wandb.log({
            'Epoch': epoch,
            'Train Loss': train_loss,
            'Train RGB Loss': train_rgb_loss,
            'Train Depth Loss': train_depth_loss,
            'Validation Loss': val_loss,
            'Validation RGB Loss': val_rgb_loss,
            'Validation Depth Loss': val_depth_loss
        })

        # Save model checkpoint if validation loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state_dict = {
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'best_val_loss': best_val_loss
            }
            safetensors.torch.save_file(state_dict, checkpoint_path)
            print(f"Saved best model checkpoint at {checkpoint_path}")

    # Finish WandB run
    wandb.finish()

if __name__ == "__main__":
    main()
