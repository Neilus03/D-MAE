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
from model.dmae import MAE, mae_loss

# Load configuration
config_path = '/home/ndelafuente/D-MAE/config/config.yaml'
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Initialize WandB
wandb.init(project=config['logging']['wandb_project'], entity=config['logging']['wandb_entity'])


def extract_patches(image, patch_size):
    '''
    Extract patches from an image tensor.
    Args:
        image: Tensor of shape (C, H, W)
        patch_size: size of the patches
    Returns:
        patches: Tensor of shape (num_patches, C, patch_size, patch_size)
    '''
    # Unfold the image to get patches
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    # Rearrange dimensions to get patches in (num_patches, channels, patch_size, patch_size)
    patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(-1, image.shape[0], patch_size, patch_size)
    return patches

def assemble_patches_with_gaps(patches, gap_size, num_patches_per_row, patch_size, num_channels=3, depth=False):
    '''
    Assembles patches into an image with gaps between patches for cooler visualization.
    Args:
        patches: numpy array of shape (num_patches, num_channels, patch_size, patch_size)
        gap_size: size of the gap between patches (in pixels)
        num_patches_per_row: number of patches per row/column
        patch_size: size of each patch (assuming square patches)
        num_channels: number of channels in the image
        depth: whether it's a depth map (True) or RGB image (False)
    Returns:
        image_with_gaps: numpy array of shape (grid_size, grid_size, num_channels) or (grid_size, grid_size)
    '''
    #define the grid size as the size of the image with gaps
    grid_size = num_patches_per_row * patch_size + (num_patches_per_row - 1) * gap_size # in pixels
    if depth:
        #for depth use a single channel
        image_with_gaps = np.ones((grid_size, grid_size))
    else:
        #for RGB use 3 channels
        image_with_gaps = np.ones((grid_size, grid_size, num_channels))
    idx = 0
    #iterate over the patches and place them in the image with gaps
    for row in range(num_patches_per_row):
        for col in range(num_patches_per_row):
            #calculate the start position of the patch
            y_start = row * (patch_size + gap_size)
            x_start = col * (patch_size + gap_size)
            if depth:
                #for depth maps we only have one channel
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size] = patches[idx]
            else:
                #for RGB images we have 3 channels
                image_with_gaps[y_start:y_start+patch_size, x_start:x_start+patch_size, :] = patches[idx].transpose(1, 2, 0)
            idx += 1
    return image_with_gaps

def log_visualizations(image_depths, reconstructed_image, reconstructed_depth, mask, epoch, batch_idx, prefix='Train'):
    '''
    Logs visualizations to WandB.
    Args:
        image_depths: input images with depth (batch_size, 4, H, W)
        reconstructed_image: reconstructed RGB images (batch_size, 3, H, W)
        reconstructed_depth: reconstructed depth maps (batch_size, 1, H, W)
        mask: mask used during training (batch_size, num_patches)
        epoch: current epoch
        batch_idx: current batch index
        prefix: 'Train' or 'Validation'
    '''
    depth_mean = config['data']['depth_stats']['mean']
    depth_std = config['data']['depth_stats']['std']
    img_size = config['model']['image_size']
    patch_size = config['model']['patch_size']

    # Select the first sample in the batch
    original_image_depth = image_depths[0].detach().cpu()  # Shape: (4, H, W)

    # Denormalize original RGB image
    original_rgb_image = denormalize_RGB(original_image_depth[:3]).permute(1, 2, 0).clamp(0, 1).numpy()
    # Denormalize original depth map
    original_depth_map = original_image_depth[3].numpy()
    original_depth_map = original_depth_map * depth_std + depth_mean

    # Normalize depth map for visualization
    original_depth_map_viz = (original_depth_map - original_depth_map.min()) / (original_depth_map.max() - original_depth_map.min() + 1e-8)

    # Extract patches from the original image
    original_patches = extract_patches(original_image_depth, patch_size)  # Shape: (num_patches, 4, patch_size, patch_size)
    original_patches_rgb = original_patches[:, :3, :, :]
    original_patches_depth = original_patches[:, 3:, :, :]

    # Denormalize patches
    original_patches_rgb_denorm = denormalize_RGB(original_patches_rgb).clamp(0, 1).numpy()  # Shape: (num_patches, 3, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth.numpy()
    original_patches_depth_denorm = original_patches_depth_denorm * depth_std + depth_mean  # Shape: (num_patches, 1, patch_size, patch_size)
    original_patches_depth_denorm = original_patches_depth_denorm.squeeze(1)  # Shape: (num_patches, patch_size, patch_size)

    # Normalize depth patches for visualization
    depth_min = original_patches_depth_denorm.min()
    depth_max = original_patches_depth_denorm.max()
    original_patches_depth_viz = (original_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble original patches with gaps
    gap_size = 2  # pixels
    num_patches_per_row = img_size[0] // patch_size
    assembled_original_rgb = assemble_patches_with_gaps(original_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_original_depth = assemble_patches_with_gaps(original_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Masked patches
    masked_patches = original_patches.clone()
    masked_indices = (mask[0] == 1).nonzero(as_tuple=False).squeeze()
    masked_patches[masked_indices] = 0  # Set masked patches to zero

    # Extract masked patches for rgb and depth separately
    masked_patches_rgb = masked_patches[:, :3, :, :] # Shape: (num_patches, 3, patch_size, patch_size)
    masked_patches_depth = masked_patches[:, 3:, :, :] # Shape: (num_patches, 1, patch_size, patch_size)

    # Denormalize masked patches
    masked_patches_rgb_denorm = denormalize_RGB(masked_patches_rgb).clamp(0, 1).numpy()
    masked_patches_depth_denorm = masked_patches_depth.numpy()
    # Denormalize masked depth patches
    masked_patches_depth_denorm = denormalize_depth(masked_patches_depth, depth_mean, depth_std)
    masked_patches_depth_denorm = masked_patches_depth_denorm.squeeze(1)

    # Normalize masked depth patches for visualization
    depth_min = masked_patches_depth_denorm.min()
    depth_max = masked_patches_depth_denorm.max()
    masked_patches_depth_viz = (masked_patches_depth_denorm - depth_min) / (depth_max - depth_min + 1e-8)

    # Assemble masked patches with gaps
    assembled_masked_rgb = assemble_patches_with_gaps(masked_patches_rgb_denorm, gap_size, num_patches_per_row, patch_size, num_channels=3)
    assembled_masked_depth = assemble_patches_with_gaps(masked_patches_depth_viz, gap_size, num_patches_per_row, patch_size, depth=True)

    # Reconstructed images
    reconstructed_rgb_denorm = denormalize_RGB(reconstructed_image[0].detach().cpu()).permute(1, 2, 0).clamp(0, 1).numpy()
    reconstructed_depth_map = reconstructed_depth[0, 0].detach().cpu().numpy()
    reconstructed_depth_map = reconstructed_depth_map * depth_std + depth_mean

    # Normalize reconstructed depth map for visualization
    recon_depth_min = reconstructed_depth_map.min()
    recon_depth_max = reconstructed_depth_map.max()
    reconstructed_depth_map_viz = (reconstructed_depth_map - recon_depth_min) / (recon_depth_max - recon_depth_min + 1e-8)

    # Create depth images using matplotlib and save them to buffers
    depth_images = {}
    # Original Depth Map
    fig1 = plt.figure()
    plt.imshow(original_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf1 = io.BytesIO()
    plt.savefig(buf1, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig1)
    buf1.seek(0)
    depth_images[f'{prefix} Original Depth Map'] = wandb.Image(Image.open(buf1), caption='Original Depth Map')

    # Assembled Original Depth Patches
    fig2 = plt.figure()
    plt.imshow(assembled_original_depth, cmap='viridis')
    plt.axis('off')
    buf2 = io.BytesIO()
    plt.savefig(buf2, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig2)
    buf2.seek(0)
    depth_images[f'{prefix} Assembled Original Depth Patches'] = wandb.Image(Image.open(buf2), caption='Original Depth Patches')

    # Masked Depth Map
    fig3 = plt.figure()
    plt.imshow(assembled_masked_depth, cmap='viridis')
    plt.axis('off')
    buf3 = io.BytesIO()
    plt.savefig(buf3, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig3)
    buf3.seek(0)
    depth_images[f'{prefix} Masked Depth Map'] = wandb.Image(Image.open(buf3), caption='Masked Depth Map')

    # Reconstructed Depth Map
    fig4 = plt.figure()
    plt.imshow(reconstructed_depth_map_viz, cmap='viridis')
    plt.axis('off')
    buf4 = io.BytesIO()
    plt.savefig(buf4, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig4)
    buf4.seek(0)
    depth_images[f'{prefix} Reconstructed Depth Map'] = wandb.Image(Image.open(buf4), caption='Reconstructed Depth Map')

    # Log images to WandB
    wandb.log({
        f'{prefix} Original RGB Image': wandb.Image(original_rgb_image),
        f'{prefix} Assembled Original RGB Patches': wandb.Image(assembled_original_rgb),
        f'{prefix} Masked RGB Image': wandb.Image(assembled_masked_rgb),
        f'{prefix} Reconstructed RGB Image': wandb.Image(reconstructed_rgb_denorm),
        **depth_images,
        'Epoch': epoch,
        'Batch': batch_idx
    })

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
